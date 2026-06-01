const camera = { azimuth: 135, pitch: 15, distance_cm: 45 };
const llmCamera = { direction: "SW", elevation_layer: 9, distance: "Full", target_layer: null, target_color: null };
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
  odd: ["Blue", "Green", "Red"],
  even: ["Blue", "Green", "Red"],
};
const faceOptions = { odd: ["North", "South"], even: ["East", "West"] };
const contactOptions = ["center", "left", "right"];

function syncPills(groupId, selectId) {
  const val = document.getElementById(selectId)?.value;
  document.querySelectorAll(`#${groupId} .pill`).forEach(p => {
    p.classList.toggle('active', p.dataset.val === val);
  });
}

function firstEnabledOption(select) {
  return Array.from(select.options).find((option) => !option.disabled)?.value || "";
}

function setPillEnabled(groupId, allowedValues) {
  const allowed = new Set(allowedValues);
  document.querySelectorAll(`#${groupId} .pill`).forEach((pill) => {
    const enabled = allowed.has(pill.dataset.val);
    pill.disabled = !enabled;
    pill.classList.toggle("disabled", !enabled);
    if (!enabled) {
      pill.classList.remove("active");
    }
  });
}

function updatePlacementOptions() {
  const select = document.querySelector("#place-position");
  const available = scene?.available_placement_positions || [];
  rebuildPills("placement-pills", available);
  if (typeof bindPills === "function") bindPills("placement-pills", "place-position", () => updateActionControls());
  const emptyOpt = new Option("", "", false, false);
  select.replaceChildren(
    emptyOpt,
    ...available.map((value) => new Option(value, value, false, false)),
  );
  if (select.value && !available.includes(select.value)) {
    select.value = "";
  }
  setPillEnabled("placement-pills", available);
  syncPills("placement-pills", "place-position");
  return available.length > 0;
}

function rebuildPills(groupId, values) {
  const group = document.querySelector(`#${groupId}`);
  group.innerHTML = values.map((value) => (
    `<button class="pill" data-val="${value}">${value}</button>`
  )).join("");
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
  const compass = document.querySelector("#compass");
  if (compass) compass.style.transform = `rotate(${camera.azimuth}deg)`;
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

function invert4x4(m) {
  const inv = new Array(16);
  inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
  inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
  inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
  inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
  inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
  inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
  inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
  inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
  inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
  inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
  inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
  inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
  inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
  inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
  inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
  inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];
  const det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
  if (Math.abs(det) < 1e-10) return null;
  const invDet = 1 / det;
  return inv.map(v => v * invDet);
}

function unproject(ndcX, ndcY, ndcZ, invVP) {
  const x = invVP[0]*ndcX + invVP[4]*ndcY + invVP[8]*ndcZ  + invVP[12];
  const y = invVP[1]*ndcX + invVP[5]*ndcY + invVP[9]*ndcZ  + invVP[13];
  const z = invVP[2]*ndcX + invVP[6]*ndcY + invVP[10]*ndcZ + invVP[14];
  const w = invVP[3]*ndcX + invVP[7]*ndcY + invVP[11]*ndcZ + invVP[15];
  return [x/w, y/w, z/w];
}

function conjugateQuat(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}

function raycastBlocks(clientX, clientY) {
  if (!scene) return null;
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;

  const yaw = camera.azimuth * Math.PI / 180;
  const pitch = camera.pitch * Math.PI / 180;
  const distance = camera.distance_cm / 100;
  const eye = [
    scene.target[0] + Math.cos(pitch) * Math.sin(yaw) * distance,
    scene.target[1] - Math.cos(pitch) * Math.cos(yaw) * distance,
    scene.target[2] + Math.sin(pitch) * distance,
  ];
  const vp = multiply(
    perspective(52, canvas.width / canvas.height, .02, 3),
    lookAt(eye, scene.target),
  );
  const invVP = invert4x4(vp);
  if (!invVP) return null;

  const near = unproject(ndcX, ndcY, -1, invVP);
  const far = unproject(ndcX, ndcY, 1, invVP);
  const dir = normalize(subtract(far, near));

  let bestT = Infinity;
  let bestBlock = null;
  let bestFaceIdx = -1;
  let bestHitLocal = null;

  for (const block of scene.blocks) {
    const half = block.size.map(v => v / 2);
    const qInv = conjugateQuat(block.rotation || [0, 0, 0, 1]);
    const localOrigin = rotate(subtract(near, block.position), qInv);
    const localDir = rotate(dir, qInv);

    let tMin = -Infinity, tMax = Infinity;
    let entryAxis = 0;

    for (let axis = 0; axis < 3; axis++) {
      if (Math.abs(localDir[axis]) < 1e-10) {
        if (localOrigin[axis] < -half[axis] || localOrigin[axis] > half[axis]) {
          tMin = Infinity;
          break;
        }
      } else {
        let t1 = (-half[axis] - localOrigin[axis]) / localDir[axis];
        let t2 = (half[axis] - localOrigin[axis]) / localDir[axis];
        if (t1 > t2) [t1, t2] = [t2, t1];
        if (t1 > tMin) { tMin = t1; entryAxis = axis; }
        if (t2 < tMax) tMax = t2;
      }
    }

    if (tMin <= tMax && tMax > 0 && tMin < bestT) {
      const t = tMin > 0 ? tMin : 0;
      bestT = t;
      bestBlock = block;
      bestFaceIdx = entryAxis;
      bestHitLocal = localOrigin.map((o, i) => o + t * localDir[i]);
    }
  }

  if (!bestBlock) return null;

  const half = bestBlock.size.map(v => v / 2);
  const faceNames = [
    bestHitLocal[0] < 0 ? "West" : "East",
    bestHitLocal[1] < 0 ? "South" : "North",
    bestHitLocal[2] < 0 ? "Bottom" : "Top",
  ];
  const faceName = faceNames[bestFaceIdx];

  let contact = "center";
  if (bestFaceIdx === 0) {
    const rel = bestHitLocal[1] / half[1];
    contact = rel < -0.33 ? "left" : rel > 0.33 ? "right" : "center";
  } else if (bestFaceIdx === 1) {
    const rel = bestHitLocal[0] / half[0];
    contact = rel < -0.33 ? "left" : rel > 0.33 ? "right" : "center";
  }

  return { block: bestBlock, face: faceName, contact, hitLocal: bestHitLocal };
}

let hoveredBlockId = null;
let selectedBlockId = null;
let selectedPush = null;
let pendingPushReselect = null;

function updateConfirmButton() {
  const btn = document.querySelector("#confirm-push");
  if (!btn) return;
  btn.disabled = !selectedPush || sandboxBusy;
  if (selectedPush) {
    btn.textContent = `Push from ${selectedPush.face}`;
  } else {
    btn.textContent = "Select a block";
  }
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

function createLineProgram() {
  const prog = gl.createProgram();
  gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, `
    attribute vec3 position;
    attribute vec3 color;
    uniform mat4 viewProjection;
    varying vec3 lineColor;
    void main() {
      gl_Position = viewProjection * vec4(position, 1.0);
      lineColor = color;
    }
  `));
  gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec3 lineColor;
    void main() {
      gl_FragColor = vec4(lineColor, 1.0);
    }
  `));
  gl.linkProgram(prog);
  return prog;
}

const lineProgram = createLineProgram();
const depthProgram = createDepthProgram();
const mainProgram = createMainProgram();
const shadowFB = createShadowFramebuffer();
const positionBuffer = gl.createBuffer();
const colorBuffer = gl.createBuffer();
const normalBuffer = gl.createBuffer();
const edgePositionBuffer = gl.createBuffer();
const edgeColorBuffer = gl.createBuffer();
const edgeNormalBuffer = gl.createBuffer();
let edgeVertexCount = 0;

const EDGE_PAIRS = [
  [0,1],[1,2],[2,3],[3,0],
  [4,5],[5,6],[6,7],[7,4],
  [0,4],[1,5],[2,6],[3,7],
];

const OUTLINE_OFFSET = 0.0012;

function appendOutline(box, edgePositions, edgeColors, edgeNormals) {
  const [cx, cy, cz] = box.position;
  const [sx, sy, sz] = box.size.map((value) => value / 2);
  const rotation = box.rotation || [0, 0, 0, 1];
  const d = OUTLINE_OFFSET;
  const vertices = [
    [-(sx+d), -(sy+d), -(sz+d)], [(sx+d), -(sy+d), -(sz+d)], [(sx+d), (sy+d), -(sz+d)], [-(sx+d), (sy+d), -(sz+d)],
    [-(sx+d), -(sy+d), (sz+d)], [(sx+d), -(sy+d), (sz+d)], [(sx+d), (sy+d), (sz+d)], [-(sx+d), (sy+d), (sz+d)],
  ].map((v) => rotate(v, rotation).map((val, i) => val + [cx, cy, cz][i]));
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
  const rgb = box._edgeColor || [0, 0, 0];
  for (let i = 0; i < faces.length; i++) {
    for (const index of [0, 1, 2, 0, 2, 3]) {
      edgePositions.push(...vertices[faces[i][index]]);
      edgeColors.push(...rgb);
      edgeNormals.push(...faceNormals[i]);
    }
  }
}

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
  const rgb = (box._colorOverride || box.color).map((channel) => channel / 255);
  const tint = box._tint || null;
  const finalRgb = tint
    ? rgb.map((c, i) => Math.min(1, c * (1 - tint.strength) + tint.color[i] * tint.strength))
    : rgb;
  for (let i = 0; i < faces.length; i++) {
    for (const index of [0, 1, 2, 0, 2, 3]) {
      positions.push(...vertices[faces[i][index]]);
      colors.push(...finalRgb);
      normals.push(...faceNormals[i]);
    }
  }
}

const ORANGE_TINT = [245/255, 166/255, 83/255];
const EDGE_COLOR = [134/255, 249/255, 255/255];

function loadGeometry() {
  const positions = [];
  const colors = [];
  const normals = [];
  const isHuman = typeof currentMode === "undefined" || currentMode === "human";
  for (const block of scene.blocks) {
    block._tint = null;
    block._tint = null;
  }
  const boxes = [scene.base, ...scene.blocks];
  if (scene.floor) boxes.unshift(scene.floor);
  const edgePositions = [];
  const edgeColors = [];
  const edgeNormals = [];
  const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
  for (const box of boxes) {
    if (box === scene.floor) {
      box._colorOverride = [150, 99, 66];
    } else if (box === scene.base) {
      box._colorOverride = [89, 56, 36];
    } else {
      box._colorOverride = null;
    }
    appendBox(box, positions, colors, normals);
    if (isHuman && (box.id === selectedBlockId || box.id === hoveredBlockId)) {
      box._edgeColor = EDGE_COLOR;
      appendOutline(box, edgePositions, edgeColors, edgeNormals);
    }
  }
  vertexCount = positions.length / 3;
  edgeVertexCount = edgePositions.length / 3;
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, edgePositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(edgePositions), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, edgeColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(edgeColors), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, edgeNormalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(edgeNormals), gl.STATIC_DRAW);
}

function bindPositions(prog) {
  const pos = gl.getAttribLocation(prog, "position");
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(pos);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const width = Math.round(rect.width * devicePixelRatio) || 512;
  const height = Math.round(rect.height * devicePixelRatio) || 512;
  if (canvas.width === width && canvas.height === height) return false;
  canvas.width = width;
  canvas.height = height;
  return true;
}

function renderScene() {
  if (!scene) return;
  resizeCanvas();
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
  const isDarkMode = document.documentElement.getAttribute('data-theme') !== 'light';
  gl.clearColor(1.0, 1.0, 1.0, 1);
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

  if (edgeVertexCount > 0) {
    gl.cullFace(gl.FRONT);
    gl.bindBuffer(gl.ARRAY_BUFFER, edgePositionBuffer);
    gl.vertexAttribPointer(gl.getAttribLocation(mainProgram, "position"), 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, edgeColorBuffer);
    gl.vertexAttribPointer(color, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, edgeNormalBuffer);
    gl.vertexAttribPointer(normal, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, edgeVertexCount);
    gl.cullFace(gl.BACK);
  }
}

new ResizeObserver(() => {
  if (resizeCanvas() && scene) renderScene();
}).observe(viewport);

function applyScene(nextScene, { preserveCamera = false } = {}) {
  scene = nextScene;
  document.querySelector("#phase").textContent = nextScene.phase;
  setSeedValue(nextScene.seed ?? 0);
  if (!preserveCamera) Object.assign(camera, nextScene.camera);
  document.querySelector("#push-layer").max = String(nextScene.max_push_layer ?? 17);
  updatePlacementOptions();
  updatePushOptions();
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
  const isLLM = typeof currentMode !== "undefined" && currentMode === "llm";
  const hint = document.querySelector("#viewport-hint");
  const pybulletImg = document.querySelector("#pybullet-frame");
  if (busy && isLLM) {
    camera.azimuth = DIRECTION_AZIMUTHS[llmCamera.direction] ?? 135;
    camera.distance_cm = DISTANCE_CM[llmCamera.distance] ?? 45;
    camera.pitch = 15;
    if (scene) scene.target = [0, 0, (llmCamera.elevation_layer - 0.5) * 0.015];
    document.querySelector("#frame").classList.remove("hidden");
    pybulletImg.classList.remove("hidden");
    pybulletImg.classList.add("pip");
    if (hint) hint.textContent = "Drag to orbit · Scroll to zoom";
  } else if (busy) {
    document.querySelector("#frame").classList.remove("hidden");
    pybulletImg.classList.add("hidden");
    pybulletImg.classList.remove("pip");
    if (hint) hint.textContent = "Drag to orbit · Scroll to zoom";
  } else if (isLLM) {
    document.querySelector("#frame").classList.add("hidden");
    pybulletImg.classList.remove("hidden");
    pybulletImg.classList.remove("pip");
    fetchPybulletFrame();
    if (hint) hint.textContent = "Locked";
  }
  updateActionControls();
}

function updateActionControls() {
  const phase = scene?.phase;
  const isLLM = typeof currentMode !== "undefined" && currentMode === "llm";
  if (isLLM || sandboxTerminated) {
    hoveredBlockId = null;
    selectedBlockId = null;
    selectedPush = null;
  } else if (phase !== "push") {
    hoveredBlockId = null;
  } else if (phase === "push" && selectedPush) {
    selectedBlockId = scene.blocks.find(b =>
      b.layer === selectedPush.layer && b.color_name === selectedPush.color
    )?.id ?? null;
  }
  updateConfirmButton();
  const wantsPush = !isLLM || (typeof llmAction !== "undefined" && llmAction === "push");
  const showPush = phase === "push" && !sandboxBusy && !sandboxTerminated && wantsPush;
  const showPlace = phase === "place_back" && !sandboxBusy && !sandboxTerminated && wantsPush;
  const showReset = !phase || sandboxTerminated || (!sandboxBusy && !showPush && !showPlace && wantsPush);
  const hasValidPushColor = Boolean(document.querySelector("#push-color").value);
  const hasValidPlacement = Boolean(document.querySelector("#place-position").value);
  document.querySelector("#panel-push").classList.toggle("hidden", !showPush);
  document.querySelector("#panel-place").classList.toggle("hidden", !showPlace);
  document.querySelector("#panel-reset").classList.toggle("hidden", !showReset);
  if (isLLM) {
    document.querySelector("#llm-action-selector").classList.toggle("hidden", sandboxTerminated);
    document.querySelector("#panel-camera").classList.toggle("hidden", sandboxTerminated || llmAction !== "viewpoint");
  }
  document.querySelector("#push").disabled = sandboxBusy || !hasValidPushColor;
  document.querySelector("#place-back").disabled = sandboxBusy || !hasValidPlacement;
  document.querySelector("#reset-tower").disabled = sandboxBusy;
  const pushLabel = document.querySelector('.llm-action-label[data-action="push"]');
  if (pushLabel) {
    pushLabel.textContent = phase === "place_back" ? "Place Back" : "Push";
  }
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
    const amount = clamp((now - started) / (1000 / 10), 0, 1);
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
  if (pendingPushReselect && !sandboxTerminated && scene?.phase === "push") {
    const p = pendingPushReselect;
    const block = scene.blocks.find(b => b.layer === p.layer && b.color_name === p.color);
    if (block) {
      selectedBlockId = block.id;
      selectedPush = { ...p };
      loadGeometry();
      renderScene();
      updateConfirmButton();
    }
  }
  pendingPushReselect = null;
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
  socket.addEventListener("close", () => {
    setStatus("Sandbox disconnected");
    if (sandboxBusy) {
      sandboxTerminated = true;
      setBusy(false);
    }
  });
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
  const layerInput = document.querySelector("#push-layer");
  const maxPushLayer = Number(scene?.max_push_layer ?? 17);
  const requestedLayer = Number(layerInput.value || 1);
  const layer = clamp(requestedLayer, 1, Math.max(maxPushLayer, 1));
  if (layer !== requestedLayer) {
    layerInput.value = String(layer);
  }
  const parity = Number(document.querySelector("#push-layer").value) % 2 ? "odd" : "even";
  const validColors = scene
    ? colorOptions[parity].filter((color) => scene.blocks.some((block) => block.layer === layer && block.color_name === color))
    : [...colorOptions[parity]];
  rebuildPills("color-pills", colorOptions[parity]);
  const colorSelect = document.querySelector("#push-color");
  colorSelect.replaceChildren(
    ...colorOptions[parity].map((value) => {
      const option = new Option(value, value, false, false);
      option.disabled = !validColors.includes(value);
      return option;
    }),
  );
  if (!validColors.includes(colorSelect.value)) {
    colorSelect.value = firstEnabledOption(colorSelect);
  }
  setPillEnabled("color-pills", validColors);

  rebuildPills("face-pills", faceOptions[parity]);
  const faceSelect = document.querySelector("#push-face");
  faceSelect.replaceChildren(...faceOptions[parity].map((value) => new Option(value)));
  if (!faceOptions[parity].includes(faceSelect.value)) {
    faceSelect.value = faceOptions[parity][0] || "";
  }
  setPillEnabled("face-pills", faceOptions[parity]);
  if (typeof bindPills === "function") {
    bindPills("color-pills", "push-color");
    bindPills("face-pills", "push-face");
  }
  syncPills("color-pills", "push-color");
  syncPills("face-pills", "push-face");
  updateActionControls();
}

function updateTargetBlockControls() {
  const targetLayerInput = document.querySelector("#cam-target-layer");
  const targetColorSelect = document.querySelector("#cam-target-color");
  const hasTarget = Boolean(targetColorSelect.value);

  targetLayerInput.disabled = !hasTarget;
  targetLayerInput.classList.toggle("is-disabled", !hasTarget);
  if (!hasTarget) {
    targetLayerInput.value = "";
  }
}

async function loadScene(path = "/api/state", method = "GET") {
  const response = await fetch(path, { method });
  if (!response.ok) throw new Error(`Scene load failed: ${response.status}`);
  applyScene(await response.json());
}

let pointerStart = null;

function canClickPush() {
  const isHuman = typeof currentMode === "undefined" || currentMode === "human";
  return isHuman && scene?.phase === "push" && !sandboxBusy && !sandboxTerminated;
}

viewport.addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, y: event.clientY };
  pointerStart = { x: event.clientX, y: event.clientY };
  viewport.setPointerCapture(event.pointerId);
});

viewport.addEventListener("pointermove", (event) => {
  if (drag) {
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    camera.azimuth = (camera.azimuth - deltaX * .55 + 360) % 360;
    camera.pitch = clamp(camera.pitch + deltaY * .4, -45, 75);
    drag = { x: event.clientX, y: event.clientY };
    updateMetadata();
    renderScene();
  } else if (canClickPush()) {
    const hit = raycastBlocks(event.clientX, event.clientY);
    const newId = hit ? hit.block.id : null;
    if (newId !== hoveredBlockId) {
      hoveredBlockId = newId;
      loadGeometry();
      renderScene();
    }
  }
});

viewport.addEventListener("pointerup", (event) => {
  const wasDrag = pointerStart && (
    Math.abs(event.clientX - pointerStart.x) > 5 ||
    Math.abs(event.clientY - pointerStart.y) > 5
  );
  drag = null;
  pointerStart = null;

  if (wasDrag || !canClickPush()) return;

  const hit = raycastBlocks(event.clientX, event.clientY);
  if (!hit) {
    if (selectedBlockId) {
      selectedBlockId = null;
      selectedPush = null;
      loadGeometry();
      renderScene();
      updateConfirmButton();
    }
    return;
  }

  const block = hit.block;
  const layer = block.layer;
  const parity = layer % 2 ? "odd" : "even";
  const validFaces = faceOptions[parity];
  let face = hit.face;
  if (!validFaces.includes(face)) {
    const yaw = camera.azimuth * Math.PI / 180;
    if (parity === "odd") {
      face = Math.cos(yaw) > 0 ? "South" : "North";
    } else {
      face = Math.sin(yaw) > 0 ? "East" : "West";
    }
  }

  if (selectedBlockId === block.id && selectedPush?.face === face) {
    selectedBlockId = null;
    selectedPush = null;
  } else {
    selectedBlockId = block.id;
    selectedPush = { layer, color: block.color_name, face };
  }
  loadGeometry();
  renderScene();
  updateConfirmButton();
});

viewport.addEventListener("pointercancel", () => { drag = null; pointerStart = null; });
viewport.addEventListener("wheel", (event) => {
  event.preventDefault();
  camera.distance_cm = clamp(camera.distance_cm + event.deltaY * .035, 20, 120);
  updateMetadata();
  renderScene();
}, { passive: false });

document.querySelector("#confirm-push").addEventListener("click", () => {
  if (!selectedPush || sandboxBusy) return;
  const contact = document.querySelector("#push-contact").value || "center";
  const intensity = document.querySelector("#push-intensity").value || "Gentle";
  document.querySelector("#sandbox-error").textContent = "";
  document.querySelector("#outcome").textContent = "-";
  const pushData = { ...selectedPush };
  pendingPushReselect = { ...selectedPush };
  selectedBlockId = null;
  selectedPush = null;
  loadGeometry();
  renderScene();
  setBusy(true);
  socket.send(JSON.stringify({
    type: "Push",
    layer: pushData.layer,
    color: pushData.color,
    face: pushData.face,
    contact: contact,
    intensity: intensity,
  }));
  updateConfirmButton();
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
  }));
});
document.querySelector("#reset-tower").addEventListener("click", () => {
  frameQueue = [];
  pendingResult = null;
  if (animation) cancelAnimationFrame(animation);
  animation = null;
  framePlaying = false;
  sandboxTerminated = false;
  hoveredBlockId = null;
  selectedBlockId = null;
  selectedPush = null;
  updateConfirmButton();
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

const DIRECTION_AZIMUTHS = { N: 0, NE: 315, E: 270, SE: 225, S: 180, SW: 135, W: 90, NW: 45 };
const DISTANCE_CM = { Close: 15, Medium: 30, Full: 45 };

document.querySelector("#change-viewpoint").addEventListener("click", () => {
  const dir = document.querySelector("#cam-direction").value || "SW";
  const elevLayer = clamp(Number(document.querySelector("#cam-elevation").value) || 9, 1, 18);
  const dist = document.querySelector("#cam-distance").value || "Full";
  const targetColor = document.querySelector("#cam-target-color").value || null;
  const targetLayer = targetColor
    ? clamp(Number(document.querySelector("#cam-target-layer").value) || elevLayer, 1, 18)
    : null;

  llmCamera.direction = dir;
  llmCamera.elevation_layer = elevLayer;
  llmCamera.distance = dist;
  llmCamera.target_layer = targetLayer;
  llmCamera.target_color = targetColor;

  updateMetadata();
  if (typeof fetchPybulletFrame === "function") fetchPybulletFrame();
});

document.querySelector("#cam-target-color").addEventListener("change", updateTargetBlockControls);
updateTargetBlockControls();
document.querySelector("#push-contact").replaceChildren(...contactOptions.map((value) => new Option(value)));
updatePushOptions();
connectSandbox();
loadScene().catch((error) => setStatus(error.message));
