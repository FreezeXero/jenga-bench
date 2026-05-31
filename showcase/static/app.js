const camera = { azimuth: 225, pitch: 15, distance_cm: 45 };
const llmCamera = { azimuth: 225, pitch: 15, distance_cm: 45, elevation_layer: 9, direction: "SW" };
const viewport = document.querySelector("#viewport");
const canvas = document.querySelector("#frame");
const loading = document.querySelector("#loading");
const status = document.querySelector("#status");
const statusDot = document.querySelector("#status-dot");
const gl = canvas.getContext("webgl", { alpha: false, antialias: true });
let scene = null;
let drag = null;
let vertexCount = 0;
let animation = null;
let socket = null;
let sandboxBusy = false;
let frameQueue = [];
let framePlaying = false;
let pendingResult = null;
let sandboxTerminated = false;
const VISUAL_POSITION_EPSILON = .0005;
const VISUAL_ROTATION_EPSILON = .002;
const colorOptions = {
  odd: ["Blue", "Brown", "Red"],
  even: ["Blue", "Brown", "Red"],
};
const faceOptions = { odd: ["North", "South"], even: ["East", "West"] };
const contactOptions = ["center", "left", "right"];

function syncPills(groupId, selectId) {
  const val = document.getElementById(selectId)?.value;
  document.querySelectorAll(`#${groupId} .pill`).forEach(p => {
    p.classList.toggle('active', p.dataset.val === val);
  });
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function activeCamera() {
  return (typeof currentMode !== "undefined" && currentMode === "llm") ? llmCamera : camera;
}

function getSeedValue() {
  return Number(document.querySelector("#tower-seed").value);
}

function setSeedValue(val) {
  document.querySelector("#tower-seed").value = String(val);
}

function updateMetadata() {
  document.querySelector("#azimuth").textContent = camera.azimuth.toFixed(2);
  document.querySelector("#pitch").textContent = camera.pitch.toFixed(2);
  document.querySelector("#distance").textContent = camera.distance_cm.toFixed(2);
}

function setStatus(text, ok = false) {
  status.textContent = text;
  statusDot.classList.toggle("ok", ok);
}

function subtract(a, b) { return a.map((value, index) => value - b[index]); }
function dot(a, b) { return a.reduce((sum, value, index) => sum + value * b[index], 0); }
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function normalize(vector) {
  const length = Math.sqrt(dot(vector, vector));
  return vector.map((value) => value / length);
}

function rotate(vector, quaternion) {
  const [x, y, z, w] = quaternion;
  const uv = cross([x, y, z], vector);
  const uuv = cross([x, y, z], uv);
  return vector.map((value, index) => value + 2 * (w * uv[index] + uuv[index]));
}

function slerp(first, second, amount) {
  let target = second;
  let cosine = dot(first, second);
  if (cosine < 0) {
    cosine = -cosine;
    target = second.map((value) => -value);
  }
  if (cosine > .9995) {
    return normalize(first.map((value, index) => value + amount * (target[index] - value)));
  }
  const angle = Math.acos(cosine);
  const sine = Math.sin(angle);
  return first.map((value, index) => (
    Math.sin((1 - amount) * angle) / sine * value
    + Math.sin(amount * angle) / sine * target[index]
  ));
}

function multiply(a, b) {
  const result = new Array(16).fill(0);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      for (let index = 0; index < 4; index += 1) {
        result[column * 4 + row] += a[index * 4 + row] * b[column * 4 + index];
      }
    }
  }
  return result;
}

function perspective(fovDegrees, aspect, near, far) {
  const focal = 1 / Math.tan(fovDegrees * Math.PI / 360);
  return [
    focal / aspect, 0, 0, 0,
    0, focal, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, 2 * far * near / (near - far), 0,
  ];
}

function lookAt(eye, target) {
  const backward = normalize(subtract(eye, target));
  const right = normalize(cross([0, 0, 1], backward));
  const up = cross(backward, right);
  return [
    right[0], up[0], backward[0], 0,
    right[1], up[1], backward[1], 0,
    right[2], up[2], backward[2], 0,
    -dot(right, eye), -dot(up, eye), -dot(backward, eye), 1,
  ];
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

const SHADOW_SIZE = 2048;
const lightDir = normalize([3.0, -4.0, 6.0]);

function createDepthProgram() {
  const prog = gl.createProgram();
  gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, `
    attribute vec3 position;
    uniform mat4 lightMatrix;
    void main() {
      gl_Position = lightMatrix * vec4(position, 1.0);
    }
  `));
  gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    void main() {
      gl_FragColor = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0);
    }
  `));
  gl.linkProgram(prog);
  return prog;
}

function createMainProgram() {
  const prog = gl.createProgram();
  gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, `
    attribute vec3 position;
    attribute vec3 color;
    attribute vec3 normal;
    uniform mat4 viewProjection;
    uniform mat4 lightMatrix;
    uniform vec3 lightDir;
    varying vec3 faceColor;
    varying vec3 faceNormal;
    varying vec4 shadowCoord;
    void main() {
      gl_Position = viewProjection * vec4(position, 1.0);
      shadowCoord = lightMatrix * vec4(position, 1.0);
      faceColor = color;
      faceNormal = normal;
    }
  `));
  gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec3 faceColor;
    varying vec3 faceNormal;
    varying vec4 shadowCoord;
    uniform sampler2D shadowMap;
    uniform vec3 lightDir;
    void main() {
      vec3 sc = shadowCoord.xyz / shadowCoord.w * 0.5 + 0.5;
      float shadow = 1.0;
      if (sc.x >= 0.0 && sc.x <= 1.0 && sc.y >= 0.0 && sc.y <= 1.0) {
        float bias = max(0.03 * (1.0 - dot(faceNormal, lightDir)), 0.008);
        float closest = texture2D(shadowMap, sc.xy).r;
        shadow = sc.z - bias > closest ? 0.4 : 1.0;
      }
      float diffuse = max(dot(faceNormal, lightDir), 0.0) * 0.6 * shadow;
      float specular = pow(max(dot(faceNormal, lightDir), 0.0), 16.0) * 0.05 * shadow;
      float ambient = 0.7;
      gl_FragColor = vec4(faceColor * min(ambient + diffuse, 1.0) + vec3(specular), 1.0);
    }
  `));
  gl.linkProgram(prog);
  return prog;
}

function createShadowFramebuffer() {
  const ext = gl.getExtension("WEBGL_depth_texture");
  if (!ext) {
    console.warn("No depth texture support — shadows disabled");
    return null;
  }
  const fb = gl.createFramebuffer();
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT, SHADOW_SIZE, SHADOW_SIZE, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, tex, 0);
  const colorTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, colorTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, SHADOW_SIZE, SHADOW_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { framebuffer: fb, depthTexture: tex };
}

function ortho(size, near, far) {
  return [
    1 / size, 0, 0, 0,
    0, 1 / size, 0, 0,
    0, 0, -2 / (far - near), 0,
    0, 0, -(far + near) / (far - near), 1,
  ];
}

function lightMatrix() {
  const target = scene ? scene.target : [0, 0, 0.135];
  const lightPos = target.map((v, i) => v + lightDir[i] * 0.8);
  return multiply(ortho(0.5, 0.01, 2.0), lookAt(lightPos, target));
}

const depthProgram = createDepthProgram();
const mainProgram = createMainProgram();
const shadowFB = createShadowFramebuffer();
const positionBuffer = gl.createBuffer();
const colorBuffer = gl.createBuffer();
const normalBuffer = gl.createBuffer();

function appendBox(box, positions, colors, normals) {
  const [cx, cy, cz] = box.position;
  const [sx, sy, sz] = box.size.map((value) => value / 2);
  const rotation = box.rotation || [0, 0, 0, 1];
  const vertices = [
    [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
    [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
  ].map((vertex) => rotate(vertex, rotation).map((value, index) => value + [cx, cy, cz][index]));
  const faceNormals = [
    [0, 0, -1], [0, 0, 1],
    [0, -1, 0], [1, 0, 0],
    [0, 1, 0], [-1, 0, 0],
  ].map((n) => rotate(n, rotation));
  const faces = [
    [0, 3, 2, 1], [4, 5, 6, 7],
    [0, 1, 5, 4], [1, 2, 6, 5],
    [2, 3, 7, 6], [3, 0, 4, 7],
  ];
  const rgb = box.color.map((channel) => channel / 255);
  for (let i = 0; i < faces.length; i++) {
    for (const index of [0, 1, 2, 0, 2, 3]) {
      positions.push(...vertices[faces[i][index]]);
      colors.push(...rgb);
      normals.push(...faceNormals[i]);
    }
  }
}

function loadGeometry() {
  const positions = [];
  const colors = [];
  const normals = [];
  const boxes = [scene.base, ...scene.blocks];
  if (scene.floor) boxes.unshift(scene.floor);
  for (const box of boxes) appendBox(box, positions, colors, normals);
  vertexCount = positions.length / 3;
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
}

function bindPositions(prog) {
  const pos = gl.getAttribLocation(prog, "position");
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(pos);
}

function renderScene() {
  if (!scene) return;
  const lm = lightMatrix();
  const yaw = camera.azimuth * Math.PI / 180;
  const pitch = camera.pitch * Math.PI / 180;
  const distance = camera.distance_cm / 100;
  const eye = [
    scene.target[0] + Math.cos(pitch) * Math.sin(yaw) * distance,
    scene.target[1] - Math.cos(pitch) * Math.cos(yaw) * distance,
    scene.target[2] + Math.sin(pitch) * distance,
  ];
  const matrix = multiply(
    perspective(52, canvas.width / canvas.height, .02, 3),
    lookAt(eye, scene.target),
  );
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  if (shadowFB) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFB.framebuffer);
    gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(depthProgram);
    gl.uniformMatrix4fv(gl.getUniformLocation(depthProgram, "lightMatrix"), false, lm);
    bindPositions(depthProgram);
    gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(mainProgram);
  gl.uniformMatrix4fv(gl.getUniformLocation(mainProgram, "viewProjection"), false, matrix);
  gl.uniformMatrix4fv(gl.getUniformLocation(mainProgram, "lightMatrix"), false, lm);
  gl.uniform3fv(gl.getUniformLocation(mainProgram, "lightDir"), lightDir);
  if (shadowFB) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, shadowFB.depthTexture);
    gl.uniform1i(gl.getUniformLocation(mainProgram, "shadowMap"), 0);
  }
  bindPositions(mainProgram);
  const color = gl.getAttribLocation(mainProgram, "color");
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.vertexAttribPointer(color, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(color);
  const normal = gl.getAttribLocation(mainProgram, "normal");
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.vertexAttribPointer(normal, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(normal);
  gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
}

function applyScene(nextScene, { preserveCamera = false } = {}) {
  scene = nextScene;
  document.querySelector("#phase").textContent = nextScene.phase;
  setSeedValue(nextScene.seed ?? 0);
  if (!preserveCamera) Object.assign(camera, nextScene.camera);
  document.querySelector("#push-layer").max = String(nextScene.max_push_layer ?? 17);
  document.querySelector("#place-position").replaceChildren(
    ...(nextScene.available_placement_positions || []).map((value) => new Option(value)),
  );
  syncPills("placement-pills", "place-position");
  loadGeometry();
  updateMetadata();
  renderScene();
  updateActionControls();
  setStatus("Connected", true);
}

function setBusy(busy) {
  sandboxBusy = busy;
  document.querySelector("#panel-sim").classList.toggle("hidden", !busy);
  document.querySelector("#inspector-layout").classList.toggle("busy", busy);
  if (busy) {
    document.querySelector("#frame").classList.remove("hidden");
    document.querySelector("#pybullet-frame").classList.add("hidden");
  } else if (typeof currentMode !== "undefined" && currentMode === "llm") {
    document.querySelector("#frame").classList.add("hidden");
    document.querySelector("#pybullet-frame").classList.remove("hidden");
    if (typeof fetchPybulletFrame === "function") fetchPybulletFrame();
  }
  updateActionControls();
}

function updateActionControls() {
  const phase = scene?.phase;
  const isLLM = typeof currentMode !== "undefined" && currentMode === "llm";
  const wantsPush = !isLLM || (typeof llmAction !== "undefined" && llmAction === "push");
  const showPush = phase === "push" && !sandboxBusy && !sandboxTerminated && wantsPush;
  const showPlace = phase === "place_back" && !sandboxBusy && !sandboxTerminated && wantsPush;
  const showReset = !phase || sandboxTerminated || (!sandboxBusy && !showPush && !showPlace && wantsPush);
  document.querySelector("#panel-push").classList.toggle("hidden", !showPush);
  document.querySelector("#panel-place").classList.toggle("hidden", !showPlace);
  document.querySelector("#panel-reset").classList.toggle("hidden", !showReset);
  document.querySelector("#push").disabled = sandboxBusy;
  document.querySelector("#place-back").disabled = sandboxBusy;
  document.querySelector("#reset-tower").disabled = sandboxBusy;
}

function applyFrame(frame, onComplete) {
  if (!scene) {
    onComplete();
    return;
  }
  const targets = new Map(frame.blocks.map((block) => [block.id, block]));
  const existing = new Map(scene.blocks.map((block) => [block.id, block]));
  scene.blocks = frame.blocks.map((target) => existing.get(target.id) || {
    id: target.id,
    position: [...target.position],
    rotation: [...target.rotation],
    size: target.size,
    color: target.color,
  });
  const starts = new Map(scene.blocks.map((block) => [
    block.id,
    { position: [...block.position], rotation: [...(block.rotation || [0, 0, 0, 1])] },
  ]));
  const started = performance.now();
  if (animation) cancelAnimationFrame(animation);
  function tick(now) {
    const amount = clamp((now - started) / (1000 / 30), 0, 1);
    for (const block of scene.blocks) {
      const from = starts.get(block.id);
      const to = targets.get(block.id);
      block.position = from.position.map((value, index) => value + amount * (to.position[index] - value));
      block.rotation = slerp(from.rotation, to.rotation, amount);
      if (to.color) block.color = to.color;
    }
    loadGeometry();
    renderScene();
    if (amount < 1) {
      animation = requestAnimationFrame(tick);
    } else {
      animation = null;
      onComplete();
    }
  }
  animation = requestAnimationFrame(tick);
  document.querySelector("#phase").textContent = frame.phase;
  document.querySelector("#sim-time").textContent = Number(frame.sim_time).toFixed(2);
  document.querySelector("#frame-count").textContent = String(frame.sequence + 1);
}

function applyResult(message) {
  if (message.scene) applyScene(message.scene, { preserveCamera: true });
  sandboxTerminated = message.outcome === "collapse";
  setBusy(false);
  document.querySelector("#outcome").textContent = message.outcome;
  document.querySelector("#frame-count").textContent = String(message.frame_count);
}

function framesVisuallyMatch(first, second) {
  if (!first || first.blocks.length !== second.blocks.length) return false;
  const previous = new Map(first.blocks.map((block) => [block.id, block]));
  return second.blocks.every((block) => {
    const before = previous.get(block.id);
    if (!before) return false;
    return block.position.every((value, index) => Math.abs(value - before.position[index]) < VISUAL_POSITION_EPSILON)
      && block.rotation.every((value, index) => Math.abs(value - before.rotation[index]) < VISUAL_ROTATION_EPSILON);
  });
}

function playQueuedFrames() {
  if (framePlaying) return;
  const frame = frameQueue.shift();
  if (!frame) {
    if (pendingResult) {
      applyResult(pendingResult);
      pendingResult = null;
    }
    return;
  }
  framePlaying = true;
  applyFrame(frame, () => {
    framePlaying = false;
    playQueuedFrames();
  });
}

function connectSandbox() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/sandbox`);
  socket.addEventListener("open", () => setStatus("Local preview", true));
  socket.addEventListener("close", () => setStatus("Sandbox disconnected"));
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    console.log("[ws]", message.type, message);
    if (message.type === "frame") {
      const previous = frameQueue[frameQueue.length - 1];
      if (message.phase === "collapse" && framesVisuallyMatch(previous, message)) {
        frameQueue[frameQueue.length - 1] = message;
      } else {
        frameQueue.push(message);
      }
      playQueuedFrames();
    }
    if (message.type === "scene") applyScene(message.scene);
    if (message.type === "result") {
      pendingResult = message;
      playQueuedFrames();
    }
    if (message.type === "error") {
      setBusy(false);
      document.querySelector("#sandbox-error").textContent = message.message;
    }
  });
}

function updatePushOptions() {
  const parity = Number(document.querySelector("#push-layer").value) % 2 ? "odd" : "even";
  document.querySelector("#push-color").replaceChildren(...colorOptions[parity].map((value) => new Option(value)));
  document.querySelector("#push-face").replaceChildren(...faceOptions[parity].map((value) => new Option(value)));
  syncPills("color-pills", "push-color");
  syncPills("face-pills", "push-face");
}

async function loadScene(path = "/api/state", method = "GET") {
  const response = await fetch(path, { method });
  if (!response.ok) throw new Error(`Scene load failed: ${response.status}`);
  applyScene(await response.json());
}

viewport.addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, y: event.clientY };
  viewport.setPointerCapture(event.pointerId);
});

viewport.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const deltaX = event.clientX - drag.x;
  const deltaY = event.clientY - drag.y;
  camera.azimuth = (camera.azimuth - deltaX * .55 + 360) % 360;
  camera.pitch = clamp(camera.pitch + deltaY * .4, -45, 75);
  drag = { x: event.clientX, y: event.clientY };
  updateMetadata();
  renderScene();
});

viewport.addEventListener("pointerup", () => { drag = null; });
viewport.addEventListener("pointercancel", () => { drag = null; });
viewport.addEventListener("wheel", (event) => {
  event.preventDefault();
  camera.distance_cm = clamp(camera.distance_cm + event.deltaY * .035, 20, 120);
  updateMetadata();
  renderScene();
}, { passive: false });

document.querySelector("#push-layer").addEventListener("input", updatePushOptions);
document.querySelector("#push").addEventListener("click", () => {
  document.querySelector("#sandbox-error").textContent = "";
  document.querySelector("#outcome").textContent = "-";
  setBusy(true);
  socket.send(JSON.stringify({
    type: "Push",
    layer: Number(document.querySelector("#push-layer").value),
    color: document.querySelector("#push-color").value,
    face: document.querySelector("#push-face").value,
    contact: document.querySelector("#push-contact").value,
    intensity: document.querySelector("#push-intensity").value,
  }));
});
document.querySelector("#place-back").addEventListener("click", () => {
  document.querySelector("#sandbox-error").textContent = "";
  document.querySelector("#outcome").textContent = "-";
  setBusy(true);
  socket.send(JSON.stringify({
    type: "PlaceBack",
    position: document.querySelector("#place-position").value,
  }));
});
document.querySelector("#reset-tower").addEventListener("click", () => {
  frameQueue = [];
  pendingResult = null;
  if (animation) cancelAnimationFrame(animation);
  animation = null;
  framePlaying = false;
  sandboxTerminated = false;
  setBusy(true);
  socket.send(JSON.stringify({
    type: "Reset",
    seed: getSeedValue(),
  }));
  setBusy(false);
  document.querySelector("#phase").textContent = "idle";
  document.querySelector("#outcome").textContent = "-";
  document.querySelector("#sim-time").textContent = "0.00";
  document.querySelector("#frame-count").textContent = "0";
});

const DIRECTION_AZIMUTHS = { N: 0, NE: 45, E: 90, SE: 135, S: 180, SW: 225, W: 270, NW: 315 };
const DISTANCE_CM = { Close: 15, Medium: 30, Full: 45 };

document.querySelector("#change-viewpoint").addEventListener("click", () => {
  const dir = document.querySelector("#cam-direction").value || "SW";
  const elevLayer = clamp(Number(document.querySelector("#cam-elevation").value) || 9, 1, 18);
  const dist = document.querySelector("#cam-distance").value || "Full";

  llmCamera.azimuth = DIRECTION_AZIMUTHS[dir] ?? 225;
  llmCamera.distance_cm = DISTANCE_CM[dist] ?? 45;
  llmCamera.elevation_layer = elevLayer;
  llmCamera.direction = dir;
  llmCamera.distance = dist;
  llmCamera.pitch = 15;

  updateMetadata();
  if (typeof fetchPybulletFrame === "function") fetchPybulletFrame();
});

document.querySelector("#push-contact").replaceChildren(...contactOptions.map((value) => new Option(value)));
updatePushOptions();
connectSandbox();
loadScene().catch((error) => setStatus(error.message));
