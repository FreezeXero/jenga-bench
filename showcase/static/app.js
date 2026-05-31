const camera = { azimuth: 225, pitch: 15, distance_cm: 45 };
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
  odd: ["Red", "Lime", "Blue"],
  even: ["Wintergreen", "Purple", "Brown"],
};
const faceOptions = { odd: ["North", "South"], even: ["East", "West"] };
const contactOptions = [
  "center",
  "top-left", "top-center", "top-right",
  "center-left", "center-right",
  "bottom-left", "bottom-center", "bottom-right",
];

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
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

function createProgram() {
  if (!gl) throw new Error("WebGL is required for the local inspector");
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, `
    attribute vec3 position;
    attribute vec3 color;
    uniform mat4 viewProjection;
    varying vec3 faceColor;
    void main() {
      gl_Position = viewProjection * vec4(position, 1.0);
      faceColor = color;
    }
  `));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec3 faceColor;
    void main() {
      gl_FragColor = vec4(faceColor, 1.0);
    }
  `));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

const program = createProgram();
const positionBuffer = gl.createBuffer();
const colorBuffer = gl.createBuffer();
const viewProjection = gl.getUniformLocation(program, "viewProjection");

function appendBox(box, positions, colors) {
  const [cx, cy, cz] = box.position;
  const [sx, sy, sz] = box.size.map((value) => value / 2);
  const rotation = box.rotation || [0, 0, 0, 1];
  const vertices = [
    [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
    [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
  ].map((vertex) => rotate(vertex, rotation).map((value, index) => value + [cx, cy, cz][index]));
  const faces = [
    [[0, 3, 2, 1], .50], [[4, 5, 6, 7], 1.05],
    [[0, 1, 5, 4], .72], [[1, 2, 6, 5], .82],
    [[2, 3, 7, 6], .66], [[3, 0, 4, 7], .76],
  ];
  for (const [indices, shade] of faces) {
    for (const index of [0, 1, 2, 0, 2, 3]) {
      positions.push(...vertices[indices[index]]);
      colors.push(...box.color.map((channel) => channel / 255 * shade));
    }
  }
}

function loadGeometry() {
  const positions = [];
  const colors = [];
  for (const box of [scene.base, ...scene.blocks]) appendBox(box, positions, colors);
  vertexCount = positions.length / 3;
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
}

function renderScene() {
  if (!scene) return;
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
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);
  gl.uniformMatrix4fv(viewProjection, false, matrix);
  const position = gl.getAttribLocation(program, "position");
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(position);
  const color = gl.getAttribLocation(program, "color");
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.vertexAttribPointer(color, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(color);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
}

function applyScene(nextScene, { preserveCamera = false } = {}) {
  scene = nextScene;
  document.querySelector("#tower-seed").value = String(nextScene.seed ?? 0);
  if (!preserveCamera) Object.assign(camera, nextScene.camera);
  document.querySelector("#push-layer").max = String(nextScene.max_push_layer ?? 17);
  document.querySelector("#place-position").replaceChildren(
    ...(nextScene.available_placement_positions || []).map((value) => new Option(value)),
  );
  loadGeometry();
  updateMetadata();
  renderScene();
  updateActionControls();
  setStatus("Local preview", true);
}

function setBusy(busy) {
  sandboxBusy = busy;
  updateActionControls();
}

function updateActionControls() {
  const placementRequired = scene?.phase === "place_back";
  document.querySelector("#push").disabled = sandboxBusy || sandboxTerminated || placementRequired;
  document.querySelector("#place-back").disabled = sandboxBusy || sandboxTerminated || !placementRequired;
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
      if (message.phase === "collapse-settle" && framesVisuallyMatch(previous, message)) {
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

document.querySelector("#reset").addEventListener("click", async () => {
  try {
    const seed = Number(document.querySelector("#tower-seed").value);
    await loadScene(`/api/reset?seed=${encodeURIComponent(seed)}`, "POST");
  } catch (error) {
    setStatus(error.message);
  }
});

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
    rotation_degrees: Number(document.querySelector("#place-rotation").value),
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
    seed: Number(document.querySelector("#tower-seed").value),
  }));
  setBusy(false);
  document.querySelector("#phase").textContent = "idle";
  document.querySelector("#outcome").textContent = "-";
  document.querySelector("#sim-time").textContent = "0.00";
  document.querySelector("#frame-count").textContent = "0";
});

document.querySelector("#capture").addEventListener("click", async () => {
  loading.classList.add("visible");
  try {
    const response = await fetch("/api/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(camera),
    });
    if (!response.ok) throw new Error(`Capture failed: ${response.status}`);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "jenga-bench-camera.png";
    link.click();
    URL.revokeObjectURL(url);
    setStatus("Authoritative PNG captured", true);
  } catch (error) {
    setStatus(error.message);
  } finally {
    loading.classList.remove("visible");
  }
});

document.querySelector("#push-contact").replaceChildren(...contactOptions.map((value) => new Option(value)));
updatePushOptions();
connectSandbox();
loadScene().catch((error) => setStatus(error.message));
