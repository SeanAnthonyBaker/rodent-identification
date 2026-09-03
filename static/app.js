/**
 * RodentWatch AI - Frontend Application
 * Handles Chronological Carousel, Battery Monitoring, Live 20s Sampling & Telemetry
 */

let detections = [];
let currentIndex = 0;
let sortOrder = "desc"; // 'desc' = newest first, 'asc' = oldest first
let sampleInterval = 60;
let nextSampleSeconds = 60;
let countdownTimerId = null;
let socket = null;

// DOM Elements
const samplingStatusText = document.getElementById("samplingStatusText");
const liveDot = document.getElementById("liveDot");
const livePulse = document.getElementById("livePulse");

// DOM Elements
const headerBatteryWidget = document.getElementById("headerBatteryWidget");
const headerBatteryValue = document.getElementById("headerBatteryValue");
const lowBatteryBanner = document.getElementById("lowBatteryBanner");
const lowBatteryPct = document.getElementById("lowBatteryPct");

const mainBatteryPct = document.getElementById("mainBatteryPct");
const batteryProgressBar = document.getElementById("batteryProgressBar");
const camDeviceName = document.getElementById("camDeviceName");
const camWifiRssi = document.getElementById("camWifiRssi");
const camTypeBadge = document.getElementById("camTypeBadge");

const inferenceEndpointLabel = document.getElementById("inferenceEndpointLabel");
const inferenceLatencyLabel = document.getElementById("inferenceLatencyLabel");
const inferenceThresholdLabel = document.getElementById("inferenceThresholdLabel");
const countdownTimer = document.getElementById("countdownTimer");
const btnSampleNow = document.getElementById("btnSampleNow");

const totalDetectionsBadge = document.getElementById("totalDetectionsBadge");
const statTotalCount = document.getElementById("statTotalCount");
const latestCaptureTime = document.getElementById("latestCaptureTime");

// Carousel Elements
const carouselEmptyState = document.getElementById("carouselEmptyState");
const carouselActiveView = document.getElementById("carouselActiveView");
const carouselMainImg = document.getElementById("carouselMainImg");
const carouselImgTimestamp = document.getElementById("carouselImgTimestamp");
const carouselImgBattery = document.getElementById("carouselImgBattery");
const carouselImgBatteryBadge = document.getElementById("carouselImgBatteryBadge");
const carouselConfidenceScore = document.getElementById("carouselConfidenceScore");
const carouselCurrentIndex = document.getElementById("carouselCurrentIndex");
const carouselTotalCount = document.getElementById("carouselTotalCount");
const carouselFrameIdx = document.getElementById("carouselFrameIdx");
const carouselEventFrameTotal = document.getElementById("carouselEventFrameTotal");
const carouselEventIdDisplay = document.getElementById("carouselEventIdDisplay");
const badgeEventStatus = document.getElementById("badgeEventStatus");
const carouselDetailFormattedTime = document.getElementById("carouselDetailFormattedTime");
const carouselDetailIsoTime = document.getElementById("carouselDetailIsoTime");
const carouselDetailDescription = document.getElementById("carouselDetailDescription");
const carouselDetailDevice = document.getElementById("carouselDetailDevice");
const btnDownloadCurrent = document.getElementById("btnDownloadCurrent");
const btnDeleteCurrent = document.getElementById("btnDeleteCurrent");
const btnDeleteEntireEvent = document.getElementById("btnDeleteEntireEvent");
const carouselPrevBtn = document.getElementById("carouselPrevBtn");
const carouselNextBtn = document.getElementById("carouselNextBtn");
const thumbnailStripContainer = document.getElementById("thumbnailStripContainer");
const thumbnailStrip = document.getElementById("thumbnailStrip");
const chkSelectAllFrames = document.getElementById("chkSelectAllFrames");
const batchActionContainer = document.getElementById("batchActionContainer");
const selectedCount = document.getElementById("selectedCount");
const btnDeleteSelectedFrames = document.getElementById("btnDeleteSelectedFrames");
let selectedFrameIds = new Set();

const sortAscBtn = document.getElementById("sortAscBtn");
const sortDescBtn = document.getElementById("sortDescBtn");
const btnRefreshGallery = document.getElementById("btnRefreshGallery");

// Live Feed Elements
const liveFeedImg = document.getElementById("liveFeedImg");
const liveFeedContainer = document.getElementById("liveFeedContainer");
const liveFrameTime = document.getElementById("liveFrameTime");
const liveFrameAnalysisStatus = document.getElementById("liveFrameAnalysisStatus");

// ROI Polygon Elements & State
const roiCanvasGarden = document.getElementById("roiCanvasGarden");
const roiCanvasCam1 = document.getElementById("roiCanvasCam1");
const roiCanvasS21 = document.getElementById("roiCanvasS21");
const roiCanvas = document.getElementById("roiCanvas"); // fallback
const btnToggleDrawZone = document.getElementById("btnToggleDrawZone");
const btnClearZone = document.getElementById("btnClearZone");
const btnToggleZoneVisibility = document.getElementById("btnToggleZoneVisibility");
const zoneVisibilityIcon = document.getElementById("zoneVisibilityIcon");
const zoneVisibilityText = document.getElementById("zoneVisibilityText");
const roiZoneBadge = document.getElementById("roiZoneBadge");
const drawInstructions = document.getElementById("drawInstructions");

const btnDrawZoneGarden = document.getElementById("btnDrawZoneGarden");
const btnClearZoneGarden = document.getElementById("btnClearZoneGarden");
const badgeZoneGarden = document.getElementById("badgeZoneGarden");
const btnDrawZoneCam1 = document.getElementById("btnDrawZoneCam1");
const btnClearZoneCam1 = document.getElementById("btnClearZoneCam1");
const badgeZoneCam1 = document.getElementById("badgeZoneCam1");
const btnDrawZoneS21 = document.getElementById("btnDrawZoneS21");
const badgeZoneS21 = document.getElementById("badgeZoneS21");

let isDrawingZone = false;
let isZoneVisible = true; // Visibility toggle for UI overlay
let currentPolygon = [];  // Points in progress [{x: 0.1, y: 0.2}, ...]
let activePolygon = [];   // Saved active zone polygon (for active camera)
let cameraPolygons = { Garden: null, cam1: null, S21: null }; // Independent per-camera zones
let currentDrawingCamera = "Garden"; // Camera being drawn on ("Garden", "cam1", or "S21")
let currentActiveCamera = "Garden";
let mousePos = null;      // Current mouse position on canvas { x, y, canvas }
let activeDrawCanvas = null;

// Zone Point Dragging State
let hoveredPointIndex = -1;
let isDraggingPoint = false;
let draggedPointIndex = -1;
let hasJustDragged = false;

// Per-Camera Digital Zoom & Focus Center State
const cameraZoomState = {
  Garden: { scale: 1.0, originX: 0.5, originY: 0.5 },
  cam1: { scale: 1.0, originX: 0.5, originY: 0.5 },
  S21: { scale: 1.0, originX: 0.5, originY: 0.5 }
};

// Target Object State & Controls
const headerTargetSelect = document.getElementById("headerTargetSelect");
const visionActiveTargetLabel = document.getElementById("visionActiveTargetLabel");
const galleryObjectFilter = document.getElementById("galleryObjectFilter");
const simTargetSelect = document.getElementById("simTargetSelect");
const btnRunSimulation = document.getElementById("btnRunSimulation");
const settingTargetObject = document.getElementById("settingTargetObject");

// Live Feed Area Target Dropdown Elements
const btnLiveTargetDropdown = document.getElementById("btnLiveTargetDropdown");
const liveTargetMenu = document.getElementById("liveTargetMenu");
const liveTargetCurrentIcon = document.getElementById("liveTargetCurrentIcon");
const liveTargetCurrentLabel = document.getElementById("liveTargetCurrentLabel");
const liveTargetCurrentSub = document.getElementById("liveTargetCurrentSub");
const liveFeedTargetDropdownContainer = document.getElementById("liveFeedTargetDropdownContainer");

let currentTargetObject = "all";
let currentFilterObject = "all";

// Settings Modal
const settingsModal = document.getElementById("settingsModal");
const btnOpenSettings = document.getElementById("btnOpenSettings");
const btnCloseSettings = document.getElementById("btnCloseSettings");
const btnCancelSettings = document.getElementById("btnCancelSettings");
const btnSaveSettings = document.getElementById("btnSaveSettings");
const settingInterval = document.getElementById("settingInterval");
const settingEngineProvider = document.getElementById("settingEngineProvider");
const containerGeminiKey = document.getElementById("containerGeminiKey");
const settingGeminiKey = document.getElementById("settingGeminiKey");
const btnToggleGeminiKeyVis = document.getElementById("btnToggleGeminiKeyVis");
const containerOllamaInputs = document.getElementById("containerOllamaInputs");
const settingEndpoint = document.getElementById("settingEndpoint");
const settingModel = document.getElementById("settingModel");
const settingThreshold = document.getElementById("settingThreshold");
const settingPhoneCamUrl = document.getElementById("settingPhoneCamUrl");

function getObjectMeta(objectType, label) {
  const ot = (objectType || "").toLowerCase();
  const lbl = (label || "").toLowerCase();
  if (ot.includes("tree") || lbl.includes("tree")) {
    return { emoji: "🌲", label: label || "Tree", color: "#22c55e", bg: "bg-emerald-600/90", border: "border-emerald-500" };
  }
  if (ot.includes("bird") || ot.includes("pheasant") || lbl.includes("bird") || lbl.includes("pheasant")) {
    const isP = ot.includes("pheasant") || lbl.includes("pheasant");
    return { emoji: isP ? "🦚" : "🐦", label: label || (isP ? "Pheasant" : "Bird"), color: "#38bdf8", bg: "bg-sky-600/90", border: "border-sky-500" };
  }
  if ((ot.includes("horse") && !ot.includes("poo")) || (lbl.includes("horse") && !lbl.includes("poo"))) {
    return { emoji: "🐴", label: label || "Horse", color: "#f59e0b", bg: "bg-amber-600/90", border: "border-amber-500" };
  }
  if (ot.includes("horses_poo") || ot.includes("poo") || ot.includes("manure") || lbl.includes("poo") || lbl.includes("manure")) {
    return { emoji: "🐴💩", label: label || "Horses poo", color: "#a855f7", bg: "bg-purple-600/90", border: "border-purple-500" };
  }
  return { emoji: "🐀", label: label || "Rat", color: "#ef4444", bg: "bg-red-600/90", border: "border-red-500" };
}

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  initRoiDrawing();
  loadSettings();
  fetchCameras();
  fetchStatus();
  fetchDetections();
  startCountdown();
  initWebSocket();
  initWebSocket();
  setRealtimeLive(true);
  selectActiveCamera("Garden");
});

// Single-Device Camera Elements
const selectActiveDevice = document.getElementById("selectActiveDevice");
const btnSelectGarden = document.getElementById("btnSelectGarden");
const btnSelectCam1 = document.getElementById("btnSelectCam1");
const btnSelectS21 = document.getElementById("btnSelectS21");

const mainSurveillanceContainer = document.getElementById("mainSurveillanceContainer");
const surveillanceTransformLayer = document.getElementById("surveillanceTransformLayer");
const mainCameraFeedImg = document.getElementById("mainCameraFeedImg");
const mainCameraRoiCanvas = document.getElementById("mainCameraRoiCanvas");

const labelActiveCameraName = document.getElementById("labelActiveCameraName");
const labelActiveCameraTimestamp = document.getElementById("labelActiveCameraTimestamp");
const badgeMainZoom = document.getElementById("badgeMainZoom");
const statusActiveCameraAnalysis = document.getElementById("statusActiveCameraAnalysis");
const btnResetMainZoom = document.getElementById("btnResetMainZoom");

// Legacy and optional layout elements safely bound
const btnViewAllZones = document.getElementById("btnViewAllZones");
const btnViewDual = document.getElementById("btnViewDual");
const btnViewGarden = document.getElementById("btnViewGarden");
const btnViewCam1 = document.getElementById("btnViewCam1");
const btnViewS21 = document.getElementById("btnViewS21");
const selectDeviceList = document.getElementById("selectDeviceList");
const btnRefreshBothFeeds = document.getElementById("btnRefreshBothFeeds");
const btnQuickFocusGarden = document.getElementById("btnQuickFocusGarden");
const btnQuickFocusCam1 = document.getElementById("btnQuickFocusCam1");
const btnSampleGarden = document.getElementById("btnSampleGarden");
const btnSampleCam1 = document.getElementById("btnSampleCam1");
const btnSampleS21 = document.getElementById("btnSampleS21");
const cardGardenTelemetry = document.getElementById("cardGardenTelemetry");
const cardCam1Telemetry = document.getElementById("cardCam1Telemetry");
const badgeGardenActive = document.getElementById("badgeGardenActive");
const badgeGardenStatus = document.getElementById("badgeGardenStatus");
const badgeCam1Active = document.getElementById("badgeCam1Active");
const badgeCam1Status = document.getElementById("badgeCam1Status");
const cameraSelectDropdown = document.getElementById("cameraSelectDropdown");

// Adjacent Zone Elements
const badgeActiveCameraZone = document.getElementById("badgeActiveCameraZone");
const textActiveCameraZone = document.getElementById("textActiveCameraZone");
const btnDrawActiveZone = document.getElementById("btnDrawActiveZone");
const textDrawActiveZone = document.getElementById("textDrawActiveZone");
const btnSaveActiveZone = document.getElementById("btnSaveActiveZone");
const btnClearActiveZone = document.getElementById("btnClearActiveZone");
const btnRotateActiveCamera = document.getElementById("btnRotateActiveCamera");
const btnToggleRealtimeLive = document.getElementById("btnToggleRealtimeLive");
const labelRealtimeLive = document.getElementById("labelRealtimeLive");

// Telemetry & Battery Badges
const headerGardenPill = document.getElementById("headerGardenPill");
const headerCam1Pill = document.getElementById("headerCam1Pill");
const headerGardenDot = document.getElementById("headerGardenDot");
const headerCam1Dot = document.getElementById("headerCam1Dot");
const headerGardenBat = document.getElementById("headerGardenBat");
const headerCam1Bat = document.getElementById("headerCam1Bat");
const gardenBatteryPct = document.getElementById("gardenBatteryPct");
const gardenBatteryBar = document.getElementById("gardenBatteryBar");
const cam1BatteryPct = document.getElementById("cam1BatteryPct");
const cam1BatteryBar = document.getElementById("cam1BatteryBar");

const savedCamera = localStorage.getItem("preferred_camera") || "S21";
let currentSelectedCamera = savedCamera; // Active camera in view: "S21", "Garden", or "cam1"
let isRealtimeLiveActive = true;
const savedRotations = JSON.parse(localStorage.getItem("camera_rotations") || "{}");
const cameraRotationState = {
  Garden: savedRotations.Garden ?? 0,
  cam1: savedRotations.cam1 ?? 0,
  S21: savedRotations.S21 ?? 0
};

async function selectActiveCamera(camName) {
  currentSelectedCamera = camName;
  currentActiveCamera = camName;

  const tabActiveEmerald = "px-3.5 py-1.5 rounded-lg font-bold bg-emerald-600 text-white shadow transition flex items-center gap-1.5";
  const tabActiveAmber = "px-3.5 py-1.5 rounded-lg font-bold bg-amber-600 text-white shadow transition flex items-center gap-1.5";
  const tabActiveBlue = "px-3.5 py-1.5 rounded-lg font-bold bg-blue-600 text-white shadow transition flex items-center gap-1.5";
  const tabInactive = "px-3.5 py-1.5 rounded-lg font-semibold text-slate-400 hover:text-white transition flex items-center gap-1.5";

  if (btnSelectGarden) btnSelectGarden.className = (camName === "Garden") ? tabActiveEmerald : tabInactive;
  if (btnSelectCam1) btnSelectCam1.className = (camName === "cam1") ? tabActiveAmber : tabInactive;
  if (btnSelectS21) btnSelectS21.className = (camName === "S21") ? tabActiveBlue : tabInactive;

  if (selectActiveDevice && selectActiveDevice.value !== camName) {
    selectActiveDevice.value = camName;
  }

  if (labelActiveCameraName) labelActiveCameraName.textContent = camName;
  updateActiveDevicePowerCard();

  // Stream update
  updateActiveCameraStream(true);

  // Reset zoom on camera switch
  surveillanceZoomState = { scale: 1.0, originX: 0.5, originY: 0.5 };
  applySurveillanceTransform();

  // Update zone status & redraw ROI canvas
  updateRoiUiState();
  resizeRoiCanvas();
  renderRoiCanvas();

  // Inform backend of primary active camera
  try {
    await fetch("/api/cameras/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_name: camName })
    });
    // Update device power card for selected camera
    updateActiveDevicePowerCard();
  } catch (e) {}
}

function updateActiveDevicePowerCard() {
  const pctEl = document.getElementById("activeDeviceBatteryPct");
  const barEl = document.getElementById("activeDeviceBatteryBar");
  const nameEl = document.getElementById("activePowerDeviceName");
  const typeEl = document.getElementById("activeDeviceCameraType");
  const iconEl = document.getElementById("activePowerIcon");

  let pct = 95;
  let devName = currentSelectedCamera;
  let typeName = "Samsung Galaxy S21 Ultra";

  if (currentSelectedCamera === "Garden") {
    pct = 92;
    devName = "Garden Camera";
    typeName = "Ring Stick Up Cam (3rd Gen)";
  } else if (currentSelectedCamera === "cam1") {
    pct = 87;
    devName = "cam1 Camera";
    typeName = "Ring Stick Up Cam (3rd Gen)";
  } else {
    pct = 95;
    devName = "Samsung S21 Ultra";
    typeName = "Samsung S21 Ultra (IP Cam)";
  }

  if (pctEl) pctEl.textContent = `${pct}%`;
  if (barEl) {
    barEl.style.width = `${pct}%`;
    barEl.className = "h-full rounded-full transition-all duration-500 " + 
      (pct <= 20 ? "bg-red-500" : pct <= 45 ? "bg-amber-500" : "bg-emerald-500");
  }
  if (nameEl) nameEl.textContent = `${devName} Power`;
  if (typeEl) typeEl.textContent = typeName;
  if (iconEl) iconEl.textContent = (pct <= 20) ? "🪫" : "🔋";
}
window.updateActiveDevicePowerCard = updateActiveDevicePowerCard;

function updateActiveCameraStream(forceReload = false) {
  if (!mainCameraFeedImg) return;
  if (isRealtimeLiveActive) {
    const targetSrc = `/api/camera/${currentSelectedCamera}/live_stream?t=${Date.now()}`;
    if (forceReload || !mainCameraFeedImg.src || !mainCameraFeedImg.src.includes(`/api/camera/${currentSelectedCamera}/live_stream`)) {
      mainCameraFeedImg.src = targetSrc;
    }
    if (labelActiveCameraTimestamp) {
      labelActiveCameraTimestamp.textContent = "Real-time Live 🔴";
    }
  } else {
    mainCameraFeedImg.src = `/api/camera/${currentSelectedCamera}/snapshot?t=${Date.now()}`;
    if (labelActiveCameraTimestamp) {
      labelActiveCameraTimestamp.textContent = `${new Date().toLocaleTimeString()} (Snapshot)`;
    }
  }
}

function initStreamWatchdog() {
  if (!mainCameraFeedImg) return;
  // Silent auto-recovery on network socket failure without flickering
  mainCameraFeedImg.onerror = () => {
    setTimeout(() => {
      if (mainCameraFeedImg && isRealtimeLiveActive) {
        mainCameraFeedImg.src = `/api/camera/${currentSelectedCamera}/live_stream?t=${Date.now()}`;
      }
    }, 1000);
  };
}

// Expose globally for inline HTML onchange handlers
window.selectActiveCamera = selectActiveCamera;
window.initStreamWatchdog = initStreamWatchdog;

function setRealtimeLive(active) {
  isRealtimeLiveActive = active;
  if (btnToggleRealtimeLive) {
    if (active) {
      btnToggleRealtimeLive.className = "px-3 py-1.5 bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-600/60 rounded-xl text-emerald-300 font-bold text-xs flex items-center gap-1.5 transition shadow";
      if (labelRealtimeLive) labelRealtimeLive.textContent = "🔴 Live";
    } else {
      btnToggleRealtimeLive.className = "px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-700 rounded-xl text-slate-400 font-semibold text-xs flex items-center gap-1.5 transition shadow";
      if (labelRealtimeLive) labelRealtimeLive.textContent = "⚪ Paused";
    }
  }
  updateActiveCameraStream();
}

function toggleRealtimeLive() {
  setRealtimeLive(!isRealtimeLiveActive);
}

// Rigid Transform Function: transforms both Image AND Canvas together with 0 pixel drift
function applySurveillanceTransform() {
  if (!surveillanceTransformLayer) return;
  const rot = cameraRotationState[currentSelectedCamera] || 0;
  const scale = surveillanceZoomState.scale || 1.0;
  const ox = (surveillanceZoomState.originX * 100).toFixed(2);
  const oy = (surveillanceZoomState.originY * 100).toFixed(2);

  surveillanceTransformLayer.style.transformOrigin = `${ox}% ${oy}%`;
  
  const transforms = [];
  if (rot !== 0) transforms.push(`rotate(${rot}deg)`);
  if (scale > 1.0) transforms.push(`scale(${scale})`);

  surveillanceTransformLayer.style.transform = transforms.length > 0 ? transforms.join(" ") : "none";
  surveillanceTransformLayer.style.transition = "transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), transform-origin 0.2s ease-out";

  if (badgeMainZoom) {
    if (scale > 1.0) {
      badgeMainZoom.textContent = `🔍 ${scale.toFixed(1)}x ZOOM`;
      badgeMainZoom.classList.remove("hidden");
    } else {
      badgeMainZoom.classList.add("hidden");
    }
  }
  if (btnResetMainZoom) {
    btnResetMainZoom.classList.toggle("hidden", scale <= 1.0);
  }
}


async function fetchCameras() {
  try {
    const res = await fetch("/api/cameras");
    if (!res.ok) return;
    const data = await res.json();
    const activeCam = (data.active_camera || "").toLowerCase();
    const cameras = data.cameras || [];

    const isGardenActive = activeCam === "garden";
    const isCam1Active = activeCam === "cam1" || activeCam === "cam 1";

    const gardenCam = cameras.find(c => c.name.toLowerCase() === "garden");
    const cam1Cam = cameras.find(c => c.name.toLowerCase() === "cam1" || c.name.toLowerCase() === "cam 1");

    // Update Garden Telemetry Tile
    if (gardenCam && gardenCam.battery_percentage !== null) {
      const gBat = gardenCam.battery_percentage;
      if (gardenBatteryPct) gardenBatteryPct.textContent = `${gBat}%`;
      if (gardenBatteryBar) gardenBatteryBar.style.width = `${gBat}%`;
      if (headerGardenBat) headerGardenBat.textContent = `${gBat}%`;
    }

    if (cardGardenTelemetry) {
      if (isGardenActive) {
        cardGardenTelemetry.className = "bg-slate-900 border-2 border-emerald-500 rounded-2xl p-4 shadow-xl flex flex-col justify-between transition ring-2 ring-emerald-500/20";
        if (badgeGardenActive) {
          badgeGardenActive.className = "text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-2 py-0.5 rounded-full font-mono font-bold flex items-center gap-1";
          badgeGardenActive.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> ACTIVE';
        }
        if (badgeGardenStatus) {
          badgeGardenStatus.className = "text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded font-mono font-bold flex items-center gap-1";
          badgeGardenStatus.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> ACTIVE';
        }
        if (headerGardenDot) headerGardenDot.className = "w-2 h-2 rounded-full bg-emerald-400 animate-pulse";
      } else {
        cardGardenTelemetry.className = "bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col justify-between transition hover:border-slate-700";
        if (badgeGardenActive) {
          badgeGardenActive.className = "text-[10px] bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded-full font-mono";
          badgeGardenActive.textContent = "STANDBY";
        }
        if (badgeGardenStatus) {
          badgeGardenStatus.className = "text-[10px] bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded font-mono";
          badgeGardenStatus.textContent = "STANDBY";
        }
        if (headerGardenDot) headerGardenDot.className = "w-2 h-2 rounded-full bg-slate-500";
      }
    }

    // Update cam1 Telemetry Tile
    if (cam1Cam && cam1Cam.battery_percentage !== null) {
      const cBat = cam1Cam.battery_percentage;
      if (cam1BatteryPct) cam1BatteryPct.textContent = `${cBat}%`;
      if (cam1BatteryBar) cam1BatteryBar.style.width = `${cBat}%`;
      if (headerCam1Bat) headerCam1Bat.textContent = `${cBat}%`;
    }

    if (cardCam1Telemetry) {
      if (isCam1Active) {
        cardCam1Telemetry.className = "bg-slate-900 border-2 border-amber-500 rounded-2xl p-4 shadow-xl flex flex-col justify-between transition ring-2 ring-amber-500/20";
        if (badgeCam1Active) {
          badgeCam1Active.className = "text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40 px-2 py-0.5 rounded-full font-mono font-bold flex items-center gap-1";
          badgeCam1Active.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span> ACTIVE';
        }
        if (badgeCam1Status) {
          badgeCam1Status.className = "text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded font-mono font-bold flex items-center gap-1";
          badgeCam1Status.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span> ACTIVE';
        }
        if (headerCam1Dot) headerCam1Dot.className = "w-2 h-2 rounded-full bg-amber-400 animate-pulse";
      } else {
        cardCam1Telemetry.className = "bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col justify-between transition hover:border-slate-700";
        if (badgeCam1Active) {
          badgeCam1Active.className = "text-[10px] bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded-full font-mono";
          badgeCam1Active.textContent = "STANDBY";
        }
        if (badgeCam1Status) {
          badgeCam1Status.className = "text-[10px] bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded font-mono";
          badgeCam1Status.textContent = "STANDBY";
        }
        if (headerCam1Dot) headerCam1Dot.className = "w-2 h-2 rounded-full bg-slate-500";
      }
    }

  } catch (err) {
    console.error("Error fetching cameras:", err);
  }
}

function initEventListeners() {
  // View mode switcher buttons & Device List
  if (btnViewAllZones) btnViewAllZones.addEventListener("click", () => setViewMode("all_zones"));
  if (btnViewDual) btnViewDual.addEventListener("click", () => setViewMode("all_zones"));
  if (btnViewGarden) btnViewGarden.addEventListener("click", () => setViewMode("Garden"));
  if (btnViewCam1) btnViewCam1.addEventListener("click", () => setViewMode("cam1"));
  if (btnViewS21) btnViewS21.addEventListener("click", () => setViewMode("S21"));

  if (selectDeviceList) {
    selectDeviceList.addEventListener("change", (e) => {
      setViewMode(e.target.value);
    });
  }

  if (btnRefreshBothFeeds) btnRefreshBothFeeds.addEventListener("click", refreshBothCameraFeeds);
  if (btnToggleRealtimeLive) btnToggleRealtimeLive.addEventListener("click", toggleRealtimeLive);

  // Quick focus buttons
  if (headerGardenPill) headerGardenPill.addEventListener("click", () => { selectCameraByName("Garden"); setViewMode("Garden"); });
  if (headerCam1Pill) headerCam1Pill.addEventListener("click", () => { selectCameraByName("cam1"); setViewMode("cam1"); });
  if (btnQuickFocusGarden) btnQuickFocusGarden.addEventListener("click", () => { selectCameraByName("Garden"); setViewMode("Garden"); });
  if (btnQuickFocusCam1) btnQuickFocusCam1.addEventListener("click", () => { selectCameraByName("cam1"); setViewMode("cam1"); });

  if (btnSampleGarden) btnSampleGarden.addEventListener("click", () => sampleCameraSpecific("Garden"));
  if (btnSampleCam1) btnSampleCam1.addEventListener("click", () => sampleCameraSpecific("cam1"));
  if (btnSampleS21) btnSampleS21.addEventListener("click", () => sampleCameraSpecific("S21"));

  // Device Dropdown & Camera Selection
  if (selectActiveDevice) {
    selectActiveDevice.addEventListener("change", (e) => {
      selectActiveCamera(e.target.value);
    });
  }
  if (btnSelectGarden) btnSelectGarden.addEventListener("click", () => selectActiveCamera("Garden"));
  if (btnSelectCam1) btnSelectCam1.addEventListener("click", () => selectActiveCamera("cam1"));
  if (btnSelectS21) btnSelectS21.addEventListener("click", () => selectActiveCamera("S21"));

  // 90° Rotation for active camera
  if (btnRotateActiveCamera) {
    btnRotateActiveCamera.addEventListener("click", async () => {
      cameraRotationState[currentSelectedCamera] = ((cameraRotationState[currentSelectedCamera] || 0) + 90) % 360;
      try {
        localStorage.setItem("camera_rotations", JSON.stringify(cameraRotationState));
      } catch (e) {}
      applySurveillanceTransform();

      if (currentSelectedCamera === "S21") {
        const orientMap = { 0: "landscape", 90: "upsidedown", 180: "upsidedown_landscape", 270: "portrait" };
        const hwOrient = orientMap[cameraRotationState.S21] || "landscape";
        try {
          await fetch(`/api/camera/S21/rotate?orientation=${hwOrient}`, {
            method: "POST"
          });
        } catch (err) {}
      }
    });
  }

function setSurveillanceZoom(zoom, originX = 0.5, originY = 0.5) {
  const z = parseFloat(zoom) || 1.0;
  surveillanceZoomState.scale = z;
  surveillanceZoomState.originX = originX;
  surveillanceZoomState.originY = originY;

  const activeClasses = "btn-main-zoom w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold border-2 border-yellow-400 ring-2 ring-yellow-400/80 bg-yellow-500/20 text-yellow-300 shadow-[0_0_8px_rgba(250,204,21,0.5)] transition";
  const inactiveClasses = "btn-main-zoom w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium border border-slate-700/60 text-slate-400 hover:text-white hover:border-slate-500 bg-slate-800/40 transition";

  document.querySelectorAll(".btn-main-zoom").forEach(b => {
    const val = parseFloat(b.dataset.zoom || b.getAttribute("data-zoom") || b.textContent);
    if (Math.abs(val - z) < 0.1) {
      b.className = activeClasses;
    } else {
      b.className = inactiveClasses;
    }
  });

  applySurveillanceTransform();
}
window.setSurveillanceZoom = setSurveillanceZoom;

  // Zoom Controls
  document.querySelectorAll(".btn-main-zoom").forEach(btn => {
    btn.addEventListener("click", () => {
      const zoom = parseFloat(btn.dataset.zoom || btn.getAttribute("data-zoom")) || 1.0;
      setSurveillanceZoom(zoom);
    });
  });

  if (btnResetMainZoom) {
    btnResetMainZoom.addEventListener("click", () => {
      setSurveillanceZoom(1.0, 0.5, 0.5);
    });
  }

  // Double click viewport to toggle 2.5x zoom into specific area
  if (mainSurveillanceContainer) {
    mainSurveillanceContainer.addEventListener("dblclick", (e) => {
      if (isDrawingZone) return;
      if (e.target.closest("button")) return;
      if (hoveredPointIndex >= 0 || isDraggingPoint) return;
      const rect = mainSurveillanceContainer.getBoundingClientRect();
      const clickX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const clickY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

      if (surveillanceZoomState.scale > 1.0) {
        surveillanceZoomState = { scale: 1.0, originX: 0.5, originY: 0.5 };
      } else {
        surveillanceZoomState = { scale: 2.5, originX: clickX, originY: clickY };
      }
      applySurveillanceTransform();
    });
  }

  // Carousel
  if (carouselPrevBtn) carouselPrevBtn.addEventListener("click", () => navigateCarousel(-1));
  if (carouselNextBtn) carouselNextBtn.addEventListener("click", () => navigateCarousel(1));
  
  // Keyboard Arrow navigation for carousel
  window.addEventListener("keydown", (e) => {
    if (["input", "textarea"].includes(document.activeElement.tagName.toLowerCase())) return;
    if (e.key === "ArrowLeft") navigateCarousel(-1);
    if (e.key === "ArrowRight") navigateCarousel(1);
  });

  // Redraw carousel bounding box on window resize
  window.addEventListener("resize", () => {
    if (detections.length > 0 && typeof currentIndex === "number") {
      drawCarouselBoundingBox(detections[currentIndex]);
    }
  });

  // Live Feed Area Target Dropdown Toggle & Options
  if (btnLiveTargetDropdown && liveTargetMenu) {
    btnLiveTargetDropdown.addEventListener("click", (e) => {
      e.stopPropagation();
      liveTargetMenu.classList.toggle("hidden");
    });

    document.querySelectorAll(".live-target-option").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const val = btn.dataset.value;
        if (val) {
          setTargetObject(val);
        }
        liveTargetMenu.classList.add("hidden");
      });
    });

    // Close on outside click
    document.addEventListener("click", (e) => {
      if (liveFeedTargetDropdownContainer && !liveFeedTargetDropdownContainer.contains(e.target)) {
        liveTargetMenu.classList.add("hidden");
      }
    });
  }

  // Target Object Quick Switch
  if (headerTargetSelect) {
    headerTargetSelect.addEventListener("change", () => {
      setTargetObject(headerTargetSelect.value);
    });
  }

  // Gallery Object Filter
  if (galleryObjectFilter) {
    galleryObjectFilter.addEventListener("change", () => {
      currentFilterObject = galleryObjectFilter.value;
      fetchDetections();
    });
  }

  // Simulation test trigger (dedicated to rat identification)
  if (btnRunSimulation) {
    btnRunSimulation.addEventListener("click", () => {
      simulateDetection("rat");
    });
  }

  // Sorting buttons
  if (sortAscBtn) sortAscBtn.addEventListener("click", () => setSortOrder("asc"));
  if (sortDescBtn) sortDescBtn.addEventListener("click", () => setSortOrder("desc"));
  if (btnRefreshGallery) btnRefreshGallery.addEventListener("click", fetchDetections);
  if (btnSampleNow) btnSampleNow.addEventListener("click", triggerSampleNow);

  const btnRotateS21 = document.getElementById("btnRotateS21");
  if (btnRotateS21) {
    btnRotateS21.addEventListener("click", async () => {
      cameraRotationState.S21 = (cameraRotationState.S21 + 90) % 360;
      applyCameraTransform("S21");

      const orientMap = { 0: "landscape", 90: "portrait", 180: "upsidedown", 270: "reverse_landscape" };
      const hwOrient = orientMap[cameraRotationState.S21] || "portrait";
      btnRotateS21.disabled = true;
      try {
        await fetch("/api/camera/S21/rotate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ orientation: hwOrient })
        });
      } catch (err) {
        console.error("Error rotating S21 hardware:", err);
      } finally {
        setTimeout(() => { btnRotateS21.disabled = false; }, 300);
      }
    });
  }

  let carouselRotationAngle = 0;
  const btnRotateCarouselImg = document.getElementById("btnRotateCarouselImg");
  if (btnRotateCarouselImg && carouselMainImg) {
    btnRotateCarouselImg.addEventListener("click", () => {
      carouselRotationAngle = (carouselRotationAngle + 90) % 360;
      carouselMainImg.style.transform = carouselRotationAngle === 0 ? "none" : `rotate(${carouselRotationAngle}deg)`;
      carouselMainImg.style.transition = "transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1)";
    });
  }

  const btnLockTargetZone = document.getElementById("btnLockTargetZone");
  const lockTargetZoneText = document.getElementById("lockTargetZoneText");
  if (btnLockTargetZone) {
    btnLockTargetZone.addEventListener("click", async () => {
      btnLockTargetZone.disabled = true;
      if (lockTargetZoneText) lockTargetZoneText.textContent = "Locking...";
      try {
        const polyArray = activePolygon && activePolygon.length >= 3
          ? activePolygon.map(p => [Number(p.x.toFixed(4)), Number(p.y.toFixed(4))])
          : null;

        const res = await fetch("/api/zone/lock_and_assess", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ polygon: polyArray })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          const isHit = data.detected;
          if (lockTargetZoneText) {
            lockTargetZoneText.textContent = isHit ? `⚠️ Baseline: ${data.label}` : "✅ Baseline Locked";
          }
          setTimeout(() => {
            if (lockTargetZoneText) lockTargetZoneText.textContent = "Lock & Assess Baseline";
            btnLockTargetZone.disabled = false;
          }, 3000);
        } else {
          if (lockTargetZoneText) lockTargetZoneText.textContent = "Lock Failed";
          setTimeout(() => {
            if (lockTargetZoneText) lockTargetZoneText.textContent = "Lock & Assess Baseline";
            btnLockTargetZone.disabled = false;
          }, 2000);
        }
      } catch (err) {
        if (lockTargetZoneText) lockTargetZoneText.textContent = "Error";
        setTimeout(() => {
          if (lockTargetZoneText) lockTargetZoneText.textContent = "Lock & Assess Baseline";
          btnLockTargetZone.disabled = false;
        }, 2000);
      }
    });
  }

  // Backlog modal
  const btnOpenBacklog = document.getElementById("btnOpenBacklog");
  const backlogModal = document.getElementById("backlogModal");
  const btnCloseBacklog = document.getElementById("btnCloseBacklog");
  const btnCloseBacklogBottom = document.getElementById("btnCloseBacklogBottom");
  if (btnOpenBacklog && backlogModal) {
    btnOpenBacklog.addEventListener("click", () => backlogModal.classList.remove("hidden"));
    if (btnCloseBacklog) btnCloseBacklog.addEventListener("click", () => backlogModal.classList.add("hidden"));
    if (btnCloseBacklogBottom) btnCloseBacklogBottom.addEventListener("click", () => backlogModal.classList.add("hidden"));
  }

  // Clear / Delete Backlog detections
  async function clearAllBacklogDetections() {
    if (!confirm("Are you sure you want to permanently delete all backlog detections and saved images? This cannot be undone.")) return;
    try {
      const res = await fetch("/api/detections/clear_all", { method: "POST" });
      if (res.ok) {
        detections = [];
        currentIndex = 0;
        await fetchDetections();
        await fetchStatus();
        if (backlogModal && !backlogModal.classList.contains("hidden")) {
          backlogModal.classList.add("hidden");
        }
      }
    } catch (e) {
      console.error("Error clearing backlog detections:", e);
    }
  }

  const btnClearGallery = document.getElementById("btnClearGallery");
  if (btnClearGallery) {
    btnClearGallery.addEventListener("click", clearAllBacklogDetections);
  }
  const btnDeleteBacklogFromModal = document.getElementById("btnDeleteBacklogFromModal");
  if (btnDeleteBacklogFromModal) {
    btnDeleteBacklogFromModal.addEventListener("click", clearAllBacklogDetections);
  }

  if (btnDeleteCurrent) {
    btnDeleteCurrent.addEventListener("click", deleteCurrentDetection);
  }
  if (btnDeleteEntireEvent) {
    btnDeleteEntireEvent.addEventListener("click", deleteEntireEvent);
  }
  if (btnDeleteSelectedFrames) {
    btnDeleteSelectedFrames.addEventListener("click", deleteSelectedFrames);
  }
  if (chkSelectAllFrames) {
    chkSelectAllFrames.addEventListener("change", (e) => {
      if (e.target.checked) {
        detections.forEach(d => selectedFrameIds.add(d.id));
      } else {
        selectedFrameIds.clear();
      }
      renderThumbnailStrip();
    });
  }

  // Settings modal
  if (btnOpenSettings && settingsModal) {
    btnOpenSettings.addEventListener("click", () => {
      settingsModal.classList.remove("hidden");
    });
    if (btnCloseSettings) btnCloseSettings.addEventListener("click", () => settingsModal.classList.add("hidden"));
    if (btnCancelSettings) btnCancelSettings.addEventListener("click", () => settingsModal.classList.add("hidden"));
    if (btnSaveSettings) btnSaveSettings.addEventListener("click", saveSettings);

    if (settingEngineProvider) {
      settingEngineProvider.addEventListener("change", () => {
        const isGemini = settingEngineProvider.value.startsWith("gemini");
        if (containerGeminiKey) containerGeminiKey.classList.toggle("hidden", !isGemini);
        if (containerOllamaInputs) containerOllamaInputs.classList.toggle("hidden", isGemini);
      });
    }

    if (btnToggleGeminiKeyVis && settingGeminiKey) {
      btnToggleGeminiKeyVis.addEventListener("click", () => {
        settingGeminiKey.type = settingGeminiKey.type === "password" ? "text" : "password";
      });
    }
  }

  window.openAndScrollSettings = function(targetElementId) {
    if (!settingsModal) return;
    settingsModal.classList.remove("hidden");
    const modalBody = settingsModal.querySelector(".overflow-y-auto");
    if (!modalBody) return;
    
    if (targetElementId) {
      const el = document.getElementById(targetElementId);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
    }
    modalBody.scrollTo({ top: modalBody.scrollHeight / 2, behavior: "smooth" });
  };

  // Close modals when clicking outside or pressing Escape
  if (settingsModal) {
    settingsModal.addEventListener("click", (e) => {
      if (e.target === settingsModal) settingsModal.classList.add("hidden");
    });
  }
  if (backlogModal) {
    backlogModal.addEventListener("click", (e) => {
      if (e.target === backlogModal) backlogModal.classList.add("hidden");
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (settingsModal) settingsModal.classList.add("hidden");
      if (backlogModal) backlogModal.classList.add("hidden");
    }
  });
}

function setSortOrder(order) {
  sortOrder = order;
  if (sortAscBtn && sortDescBtn) {
    if (order === "asc") {
      sortAscBtn.className = "px-3 py-1 rounded-lg font-medium bg-amber-600 text-white shadow";
      sortDescBtn.className = "px-3 py-1 rounded-lg font-medium text-slate-400 hover:text-slate-200";
    } else {
      sortDescBtn.className = "px-3 py-1 rounded-lg font-medium bg-amber-600 text-white shadow";
      sortAscBtn.className = "px-3 py-1 rounded-lg font-medium text-slate-400 hover:text-slate-200";
    }
  }
  fetchDetections();
}

// --- Battery Status Updater ---
function updateBatteryDisplay(battery, isMock, deviceName, wifiRssi) {
  if (battery !== null && battery !== undefined) {
    const pct = Math.round(battery);
    if (headerBatteryValue) headerBatteryValue.innerHTML = `<span class="battery-icon">🔋</span><span>${pct}%</span>`;
    if (mainBatteryPct) {
      mainBatteryPct.textContent = `${pct}%`;
      mainBatteryPct.className = "text-2xl font-bold font-mono " + 
        (pct <= 20 ? "text-red-400 animate-pulse" : pct <= 45 ? "text-amber-400" : "text-emerald-400");
    }
    if (batteryProgressBar) {
      batteryProgressBar.style.width = `${pct}%`;
      batteryProgressBar.className = "h-full rounded-full transition-all duration-500 " + 
        (pct <= 20 ? "bg-red-500" : pct <= 45 ? "bg-amber-500" : "bg-emerald-500");
    }

    // Low battery warning banner
    if (lowBatteryBanner) {
      if (pct <= 20) {
        lowBatteryBanner.classList.remove("hidden");
        if (lowBatteryPct) lowBatteryPct.textContent = pct;
      } else {
        lowBatteryBanner.classList.add("hidden");
      }
    }
  } else {
    if (headerBatteryValue) headerBatteryValue.innerHTML = `<span class="battery-icon">⚡</span><span>Wired</span>`;
    if (mainBatteryPct) mainBatteryPct.textContent = "Wired";
    if (batteryProgressBar) {
      batteryProgressBar.style.width = "100%";
      batteryProgressBar.className = "h-full rounded-full bg-blue-500";
    }
    if (lowBatteryBanner) lowBatteryBanner.classList.add("hidden");
  }

  if (deviceName && typeof camDeviceName !== "undefined" && camDeviceName) camDeviceName.textContent = deviceName;
  if (wifiRssi && typeof camWifiRssi !== "undefined" && camWifiRssi) camWifiRssi.textContent = `${wifiRssi} dBm`;
  
  if (typeof camTypeBadge !== "undefined" && camTypeBadge) {
    if (isMock) {
      camTypeBadge.textContent = "Mock / Simulation";
      camTypeBadge.className = "text-xs bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded border border-amber-500/30";
    } else {
      camTypeBadge.textContent = "Online (Ring API)";
      camTypeBadge.className = "text-xs bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30";
    }
  }
}

// --- Status API ---
async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();

    if (data.device_health) {
      updateBatteryDisplay(
        data.device_health.battery_percentage,
        data.device_health.is_mock,
        data.device_health.device_name,
        data.device_health.wifi_signal_strength
      );
    }

    if (data.latest_inference) {
      inferenceLatencyLabel.textContent = `${Math.round(data.latest_inference.inference_time_ms)} ms`;
    }

    if (data.inference_engine) {
      updateVisionEngineUi(
        data.inference_engine.endpoint_type,
        data.inference_engine.gemini_model,
        data.inference_engine.model_name
      );
    }

    if (data.storage_stats) {
      statTotalCount.textContent = data.storage_stats.total_detections;
      totalDetectionsBadge.textContent = `${data.storage_stats.total_detections} Detected`;
      if (data.storage_stats.latest_detection_timestamp) {
        latestCaptureTime.textContent = new Date(data.storage_stats.latest_detection_timestamp).toLocaleTimeString();
      }
    }

    if (data.interval_seconds) {
      sampleInterval = data.interval_seconds;
    }
  } catch (err) {
    console.error("Error fetching status:", err);
  }
}

// --- Detections & Chronological Carousel ---
async function fetchDetections() {
  try {
    const filterQuery = (currentFilterObject && currentFilterObject !== "all") ? `&object_type=${encodeURIComponent(currentFilterObject)}` : "";
    const res = await fetch(`/api/detections?order=${sortOrder}&limit=500${filterQuery}`);
    if (!res.ok) return;
    const data = await res.json();
    detections = data.detections || [];
    
    statTotalCount.textContent = data.total;
    totalDetectionsBadge.textContent = `${data.total} Identified`;

    if (detections.length === 0) {
      carouselEmptyState.classList.remove("hidden");
      carouselActiveView.classList.add("hidden");
      thumbnailStripContainer.classList.add("hidden");
      return;
    }

    carouselEmptyState.classList.add("hidden");
    carouselActiveView.classList.remove("hidden");
    thumbnailStripContainer.classList.remove("hidden");

    // Clamp current index
    if (currentIndex >= detections.length) {
      currentIndex = detections.length - 1;
    }
    if (currentIndex < 0) currentIndex = 0;

    renderCarousel();
    renderThumbnailStrip();
  } catch (err) {
    console.error("Error fetching detections:", err);
  }
}

async function setTargetObject(target) {
  currentTargetObject = target;
  if (headerTargetSelect) headerTargetSelect.value = target;
  if (settingTargetObject) settingTargetObject.value = target;
  
  if (galleryObjectFilter) {
    galleryObjectFilter.value = target;
    currentFilterObject = target;
  }

  const targetMetaMap = {
    all: { icon: "🌐", label: "All Objects", sub: "Simultaneous multi-target identification", color: "#f59e0b" },
    tree: { icon: "🌲", label: "Tree", sub: "Trees, trunks, branches & foliage", color: "#22c55e" },
    bird: { icon: "🐦", label: "Bird", sub: "Pheasants, songbirds & avian visitors", color: "#38bdf8" },
    rat: { icon: "🐀", label: "Rat", sub: "Rats, mice & rodents", color: "#ef4444" },
    horse: { icon: "🐴", label: "Horse", sub: "Live horses, ponies & equines", color: "#f59e0b" },
    horses_poo: { icon: "🐴💩", label: "Horses poo", sub: "Manure, dung piles & droppings", color: "#a855f7" }
  };

  const meta = targetMetaMap[target] || { icon: "🎯", label: target, sub: "Focused vision target", color: "#f59e0b" };

  // Update Live Feed Target Dropdown Button
  if (liveTargetCurrentIcon) liveTargetCurrentIcon.textContent = meta.icon;
  if (liveTargetCurrentLabel) liveTargetCurrentLabel.textContent = meta.label;
  if (liveTargetCurrentSub) liveTargetCurrentSub.textContent = meta.sub;

  // Highlight selected option in scrollable menu
  document.querySelectorAll(".live-target-option").forEach((btn) => {
    const isSelected = btn.dataset.value === target;
    if (isSelected) {
      btn.classList.add("bg-amber-500/20", "border-amber-500/50", "ring-1", "ring-amber-500/30");
    } else {
      btn.classList.remove("bg-amber-500/20", "border-amber-500/50", "ring-1", "ring-amber-500/30");
    }
  });

  if (visionActiveTargetLabel) {
    visionActiveTargetLabel.textContent = `${meta.icon} ${meta.label}`;
  }

  try {
    await fetch("/api/target_object", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_object: target })
    });
    await fetchDetections();
    await fetchStatus();
  } catch (e) {
    console.error("Error setting target object:", e);
  }
}


async function simulateDetection(target) {
  try {
    const res = await fetch(`/api/simulate_detection?animal=${encodeURIComponent(target || "rat")}`, {
      method: "POST"
    });
    if (res.ok) {
      await fetchDetections();
    }
  } catch (e) {
    console.error("Error simulating detection:", e);
  }
}

function renderCarousel() {
  if (detections.length === 0) return;
  const item = detections[currentIndex];

  carouselMainImg.src = item.image_url;
  carouselImgTimestamp.textContent = item.formatted_time;
  
  if (item.battery_percentage !== null) {
    carouselImgBatteryBadge.classList.remove("hidden");
    carouselImgBattery.textContent = `${item.battery_percentage}%`;
  } else {
    carouselImgBatteryBadge.classList.add("hidden");
  }

  // Update dynamic badge using getObjectMeta
  const meta = getObjectMeta(item.object_type, item.label);
  const carouselAlertBadge = document.getElementById("carouselAlertBadge");
  const carouselAlertBadgeEmoji = document.getElementById("carouselAlertBadgeEmoji");
  const carouselAlertBadgeText = document.getElementById("carouselAlertBadgeText");
  if (carouselAlertBadge) {
    carouselAlertBadge.className = `absolute top-4 right-4 ${meta.bg} backdrop-blur text-white text-xs font-bold px-3 py-1.5 rounded-lg shadow-lg flex items-center gap-1.5 border border-white/20`;
  }
  if (carouselAlertBadgeEmoji) carouselAlertBadgeEmoji.textContent = meta.emoji;
  if (carouselAlertBadgeText) carouselAlertBadgeText.innerHTML = `${meta.label.toUpperCase()} DETECTED &bull; <span id="carouselConfidenceScore">${Math.round(item.confidence * 100)}% Conf.</span>`;

  // Calculate Event Session and Frame Index
  const eventId = item.event_id || `evt_${item.id}`;
  const eventFrames = detections.filter(d => (d.event_id || `evt_${d.id}`) === eventId);
  const frameIdx = eventFrames.findIndex(d => d.id === item.id) + 1;
  const eventTotal = eventFrames.length;

  if (carouselFrameIdx) carouselFrameIdx.textContent = frameIdx || (item.frame_index || 1);
  if (carouselEventFrameTotal) carouselEventFrameTotal.textContent = eventTotal;
  if (carouselEventIdDisplay) carouselEventIdDisplay.textContent = eventId;
  
  const uniqueEvents = new Set(detections.map(d => d.event_id || `evt_${d.id}`)).size;
  if (badgeEventStatus) {
    badgeEventStatus.textContent = `${uniqueEvents} Event${uniqueEvents === 1 ? '' : 's'}`;
  }

  carouselCurrentIndex.textContent = currentIndex + 1;
  carouselTotalCount.textContent = detections.length;

  carouselDetailFormattedTime.textContent = item.formatted_time;
  carouselDetailIsoTime.textContent = item.timestamp;
  carouselDetailDescription.textContent = item.description || `${meta.label} identified in scene by Gemma 4 E12b.`;
  carouselDetailDevice.textContent = item.device_name || "Ring Camera";

  btnDownloadCurrent.href = item.image_url;
  btnDownloadCurrent.download = item.filename;

  // Highlight active thumbnail
  updateActiveThumbnail();

  // Draw bounding box overlay
  drawCarouselBoundingBox(item);
}

function drawCarouselBoundingBox(item) {
  const canvas = document.getElementById("carouselBboxCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let box = item.bounding_box;
  if (typeof box === "string") {
    try { box = JSON.parse(box); } catch (e) { box = null; }
  }

  const drawBox = () => {
    if (!canvas || !canvas.parentElement) return;
    const parent = canvas.parentElement.getBoundingClientRect();
    if (parent.width === 0 || parent.height === 0) {
      setTimeout(drawBox, 50);
      return;
    }

    canvas.width = parent.width;
    canvas.height = parent.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!box || !Array.isArray(box) || box.length < 4) return;

    // Use exact getBoundingClientRect of image if rendered, else calculate from natural dimensions
    const imgRect = carouselMainImg.getBoundingClientRect();
    let dispW, dispH, offsetX, offsetY;

    if (imgRect.width > 0 && imgRect.height > 0) {
      dispW = imgRect.width;
      dispH = imgRect.height;
      offsetX = imgRect.left - parent.left;
      offsetY = imgRect.top - parent.top;
    } else {
      const imgAspect = (carouselMainImg.naturalWidth || 16) / (carouselMainImg.naturalHeight || 9);
      const parentAspect = parent.width / parent.height;
      if (imgAspect > parentAspect) {
        dispW = parent.width;
        dispH = parent.width / imgAspect;
        offsetX = 0;
        offsetY = (parent.height - dispH) / 2;
      } else {
        dispH = parent.height;
        dispW = parent.height * imgAspect;
        offsetX = (parent.width - dispW) / 2;
        offsetY = 0;
      }
    }

    // box format: [ymin, xmin, ymax, xmax] (0-1000 or 0.0-1.0)
    const ymin = (box[0] > 1 ? box[0] / 1000 : box[0]) * dispH + offsetY;
    const xmin = (box[1] > 1 ? box[1] / 1000 : box[1]) * dispW + offsetX;
    const ymax = (box[2] > 1 ? box[2] / 1000 : box[2]) * dispH + offsetY;
    const xmax = (box[3] > 1 ? box[3] / 1000 : box[3]) * dispW + offsetX;

    // Place target horizontally centered and vertically JUST ABOVE the rat
    const cx = (xmin + xmax) / 2;
    const reticleRadius = 15;
    let cy = ymin - reticleRadius - 12;
    if (cy < offsetY + 35) {
      cy = ymin + reticleRadius + 6;
    }

    const meta = getObjectMeta(item.object_type, item.label);
    drawTargetReticle(ctx, cx, cy, meta.label, item.object_type, item.confidence, meta.color, meta.emoji, ymin);
  };

  drawBox();
  if (!carouselMainImg.complete || carouselMainImg.naturalWidth === 0) {
    carouselMainImg.addEventListener("load", drawBox, { once: true });
  }
  setTimeout(drawBox, 60);

  setTimeout(drawBox, 250);
}

async function deleteSpecificDetection(id, idx) {
  if (!confirm(`Delete frame #${id}?`)) return;
  try {
    const res = await fetch(`/api/detections/${id}`, { method: "DELETE" });
    if (res.ok) {
      selectedFrameIds.delete(id);
      if (currentIndex >= idx && currentIndex > 0) {
        currentIndex--;
      }
      await fetchDetections();
      await fetchStatus();
    }
  } catch (err) {
    console.error("Error deleting detection:", err);
  }
}

async function deleteEntireEvent() {
  if (detections.length === 0) return;
  const item = detections[currentIndex];
  const eventId = item.event_id || `evt_${item.id}`;
  const eventFrames = detections.filter(d => (d.event_id || `evt_${d.id}`) === eventId);
  const count = eventFrames.length;

  if (!confirm(`Delete entire Event session (${count} frame${count === 1 ? '' : 's'})? This will delete ALL frames for this sighting.`)) return;

  try {
    const res = await fetch(`/api/events/${encodeURIComponent(eventId)}`, { method: "DELETE" });
    if (res.ok) {
      eventFrames.forEach(f => selectedFrameIds.delete(f.id));
      await fetchDetections();
      await fetchStatus();
    }
  } catch (err) {
    console.error("Error deleting event:", err);
  }
}

async function deleteSelectedFrames() {
  if (selectedFrameIds.size === 0) return;
  const count = selectedFrameIds.size;
  if (!confirm(`Delete ${count} selected frame${count === 1 ? '' : 's'}?`)) return;

  const ids = Array.from(selectedFrameIds);
  try {
    const res = await fetch("/api/detections/delete_batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: ids })
    });
    if (res.ok) {
      selectedFrameIds.clear();
      await fetchDetections();
      await fetchStatus();
    }
  } catch (err) {
    console.error("Error deleting selected frames:", err);
  }
}

function updateBatchActionUi() {
  const count = selectedFrameIds.size;
  if (selectedCount) selectedCount.textContent = count;
  if (batchActionContainer) {
    batchActionContainer.classList.toggle("hidden", count === 0);
  }
  if (chkSelectAllFrames) {
    chkSelectAllFrames.checked = detections.length > 0 && selectedFrameIds.size === detections.length;
    chkSelectAllFrames.indeterminate = selectedFrameIds.size > 0 && selectedFrameIds.size < detections.length;
  }
}

function renderThumbnailStrip() {
  thumbnailStrip.innerHTML = "";
  detections.forEach((det, idx) => {
    const thumb = document.createElement("div");
    const isSelected = selectedFrameIds.has(det.id);
    thumb.className = `thumbnail-item flex-shrink-0 w-24 h-16 rounded-xl overflow-hidden border-2 bg-black cursor-pointer relative group transition ${
      idx === currentIndex ? "border-amber-500 ring-2 ring-amber-500/50 shadow-lg" : "border-slate-800 hover:border-slate-600"
    }`;
    thumb.innerHTML = `
      <img src="${det.image_url}" alt="Capture ${idx + 1}" class="w-full h-full object-cover" />
      
      <!-- Selection Checkbox -->
      <label class="absolute top-1 left-1 z-20 cursor-pointer bg-black/60 p-0.5 rounded backdrop-blur" title="Select Frame #${det.id}">
        <input type="checkbox" class="chk-frame-select w-3.5 h-3.5 rounded bg-slate-900 border-slate-600 text-amber-500 cursor-pointer" data-id="${det.id}" ${isSelected ? "checked" : ""} />
      </label>

      <!-- Frame Index Badge -->
      <span class="absolute bottom-1 left-1 text-[9px] font-mono bg-black/80 text-amber-400 px-1 rounded backdrop-blur">#${det.id} (F${det.frame_index || 1})</span>

      <!-- Delete Frame button on hover -->
      <button type="button" class="btn-delete-thumb absolute top-1 right-1 bg-red-700/90 hover:bg-red-600 text-white rounded p-1 opacity-0 group-hover:opacity-100 transition z-20" title="Delete Frame #${det.id}">
        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M6 18L18 6M6 6l12 12"></path></svg>
      </button>
    `;

    const chk = thumb.querySelector(".chk-frame-select");
    if (chk) {
      chk.addEventListener("click", (e) => {
        e.stopPropagation();
      });
      chk.addEventListener("change", (e) => {
        if (e.target.checked) {
          selectedFrameIds.add(det.id);
        } else {
          selectedFrameIds.delete(det.id);
        }
        updateBatchActionUi();
      });
    }

    thumb.addEventListener("click", (e) => {
      if (e.target.closest(".btn-delete-thumb")) {
        e.stopPropagation();
        deleteSpecificDetection(det.id, idx);
        return;
      }
      currentIndex = idx;
      renderCarousel();
    });
    thumbnailStrip.appendChild(thumb);
  });
  updateBatchActionUi();
}

function updateActiveThumbnail() {
  const thumbs = thumbnailStrip.querySelectorAll(".thumbnail-item");
  thumbs.forEach((t, idx) => {
    if (idx === currentIndex) {
      t.classList.add("border-amber-500", "ring-2", "ring-amber-500/50");
      t.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    } else {
      t.classList.remove("border-amber-500", "ring-2", "ring-amber-500/50");
    }
  });
}

function navigateCarousel(delta) {
  if (detections.length === 0) return;
  currentIndex = (currentIndex + delta + detections.length) % detections.length;
  renderCarousel();
}

async function deleteCurrentDetection() {
  if (detections.length === 0) return;
  const item = detections[currentIndex];
  if (!confirm(`Delete frame #${item.id} captured on ${item.formatted_time}?`)) return;

  try {
    const res = await fetch(`/api/detections/${item.id}`, { method: "DELETE" });
    if (res.ok) {
      selectedFrameIds.delete(item.id);
      detections.splice(currentIndex, 1);
      if (currentIndex >= detections.length) {
        currentIndex = Math.max(0, detections.length - 1);
      }
      await fetchDetections();
      await fetchStatus();
    }
  } catch (err) {
    console.error("Error deleting detection:", err);
  }
}

// --- Sample Countdown & Manual Trigger ---
let lastLiveVideoFrameTime = 0;

function startCountdown() {
  if (countdownTimerId) clearInterval(countdownTimerId);
  if (countdownTimer) {
    countdownTimer.textContent = "⚡ Instant Delta";
  }
}

async function triggerSampleNow() {
  btnSampleNow.disabled = true;
  btnSampleNow.textContent = "Sampling...";
  try {
    const res = await fetch("/api/sample_now", { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      if (data.latest_image_base64 && (Date.now() - lastLiveVideoFrameTime > 3000)) {
        liveFeedImg.src = data.latest_image_base64;
      }
      liveFrameTime.textContent = data.timestamp;

      const isHit = data.detected || data.rat_detected;
      const isPheasant = (data.object_type || "").toLowerCase() === "pheasant";
      const tagEmoji = isPheasant ? "🦚" : "🚨";
      const tagLabel = data.label || (isPheasant ? "Pheasant" : "Rat");

      if (isHit) {
        lastDetectedTargetBox = data.bounding_box;
        lastDetectedTargetTime = Date.now();
        lastDetectedTargetLabel = tagLabel;
        lastDetectedTargetType = data.object_type || (isPheasant ? "pheasant" : "rat");
        lastDetectedTargetConf = data.confidence || 0.0;
        renderRoiCanvas();
      }

      liveFrameAnalysisStatus.textContent = isHit 
        ? `${tagEmoji} ${tagLabel} Identified (${Math.round(data.confidence * 100)}%)` 
        : `Clear scene (${Math.round(data.confidence * 100)}%)`;

      if (data.detection_saved) {
        fetchDetections();
      }
    }
  } catch (err) {
    console.error("Error triggering sample:", err);
  } finally {
    btnSampleNow.disabled = false;
    btnSampleNow.textContent = "⚡ Sample Now";
  }
}

// --- WebSocket Live Stream ---
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  socket = new WebSocket(wsUrl);

  socket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "live_video_frame") {
        lastLiveVideoFrameTime = Date.now();
        const payload = msg.data;
        if (payload.image_base64 && liveFeedImg) {
          liveFeedImg.src = payload.image_base64;
          if (liveFrameTime) liveFrameTime.textContent = `${payload.timestamp} (Live Stream 🔴)`;

          // Real-time Target Zone Delta meter
          const zoneDeltaVal = document.getElementById("zoneDeltaVal");
          const targetZoneDeltaBadge = document.getElementById("targetZoneDeltaBadge");
          if (zoneDeltaVal && typeof payload.zone_delta_pct === "number") {
            const isDelta = payload.has_material_delta;
            zoneDeltaVal.textContent = isDelta 
              ? `${payload.zone_delta_pct.toFixed(1)}% (⚡ Delta Triggered)` 
              : `${payload.zone_delta_pct.toFixed(1)}% (Quiet)`;

            if (targetZoneDeltaBadge) {
              if (isDelta) {
                targetZoneDeltaBadge.className = "text-xs bg-amber-500/20 text-amber-300 border border-amber-500/50 px-2.5 py-1 rounded-full font-mono flex items-center gap-1.5 animate-pulse";
              } else {
                targetZoneDeltaBadge.className = "text-xs bg-slate-800 text-slate-300 border border-slate-700 px-2.5 py-1 rounded-full font-mono flex items-center gap-1.5";
              }
            }
          }

          renderRoiCanvas();
        }
        return;
      }
      if (msg.type === "target_object_changed") {
        if (msg.data && msg.data.target_object) {
          const t = msg.data.target_object;
          currentTargetObject = t;
          if (headerTargetSelect) headerTargetSelect.value = t;
          if (settingTargetObject) settingTargetObject.value = t;
          if (visionActiveTargetLabel) {
            const labels = {
              all: "🌐 All Objects",
              rat: "🐀 Rat",
              bird: "🐦 Bird",
              tree: "🌲 Tree",
              horse: "🐴 Horse",
              horses_poo: "🐴💩 Horses poo"
            };
            visionActiveTargetLabel.textContent = labels[t] || t;
          }
        }
        return;
      }

      if (msg.type === "camera_switched") {
        fetchCameras();
        fetchStatus();
        return;
      }

      if (msg.type === "possible") {
        const payload = msg.data;
        if (liveFrameAnalysisStatus) {
          liveFrameAnalysisStatus.textContent = `⚡ Possible Rat (${Math.round(payload.confidence * 100)}%) — Validating on Spark...`;
        }
        if (payload.bounding_box) {
          lastDetectedTargetBox = payload.bounding_box;
          lastDetectedTargetTime = Date.now();
          lastDetectedTargetLabel = "Possible Rat";
          lastDetectedTargetType = "rat";
          lastDetectedTargetConf = payload.confidence;
          renderRoiCanvas();
        }
        return;
      }

      if (msg.type === "verdict") {
        const payload = msg.data;
        const loc = payload.location || "Garden run";
        if (liveFrameAnalysisStatus) {
          liveFrameAnalysisStatus.textContent = `🚨 RAT CONFIRMED (${Math.round(payload.confidence * 100)}%) — ${loc} [Spark: ${payload.spark_ms}ms]`;
        }
        if (payload.image_base64 && liveFeedImg) {
          liveFeedImg.src = payload.image_base64;
        }
        lastDetectedTargetTime = Date.now();
        lastDetectedTargetLabel = `Rat (${loc})`;
        lastDetectedTargetType = "rat";
        lastDetectedTargetConf = payload.confidence;
        renderRoiCanvas();
        fetchDetections();
        return;
      }

      if (msg.type === "rejected") {
        const payload = msg.data;
        if (liveFrameAnalysisStatus) {
          liveFrameAnalysisStatus.textContent = `⚪ Rejected: ${payload.reason || "Not a rat"} [Spark: ${payload.spark_ms}ms]`;
        }
        lastDetectedTargetBox = null;
        renderRoiCanvas();
        return;
      }

      // Stage 1: Object detected in camera zone -> Switch to real-time and draw object boundary ONLY IF IT COULD BE AN ANIMAL
      if (msg.type === "object_detected") {
        const payload = msg.data;
        if (payload.interval_seconds) {
          sampleInterval = payload.interval_seconds;
        }
        // Only bound if candidate could be an animal
        if (payload.object_boundary && payload.is_animal !== false) {
          lastDetectedTargetBox = payload.object_boundary;
          lastDetectedTargetTime = Date.now();
          if (!lastDetectedTargetLabel || lastDetectedTargetLabel === "Detecting Animal...") {
            lastDetectedTargetLabel = "Detecting Animal...";
            lastDetectedTargetType = "candidate_animal";
            lastDetectedTargetConf = null;
          }
          renderRoiCanvas();
        } else if (!payload.object_boundary && (Date.now() - lastDetectedTargetTime > 1500)) {
          lastDetectedTargetBox = null;
          renderRoiCanvas();
        }
        if (liveFrameAnalysisStatus) {
          liveFrameAnalysisStatus.textContent = payload.is_animal !== false
            ? `⚡ Potential Animal in Zone — Real-Time Active (${payload.interval_seconds || 1}s)`
            : `⚡ Zone Activity (${(payload.delta_percent || 0).toFixed(1)}% delta) — Real-Time Active (${payload.interval_seconds || 1}s)`;
        }
        if (samplingStatusText) {
          samplingStatusText.textContent = `⚡ Real-Time Active (${payload.interval_seconds || 1}s)`;
          samplingStatusText.className = "text-xs font-bold text-amber-400 animate-pulse";
        }
        return;
      }

      // Cadence reversion after quiet cooldown
      if (msg.type === "cadence_changed") {
        const payload = msg.data;
        if (payload.interval_seconds) {
          sampleInterval = payload.interval_seconds;
        }
        if (payload.sampling_cadence === "idle") {
          lastDetectedTargetBox = null;
          if (liveFrameAnalysisStatus) {
            liveFrameAnalysisStatus.textContent = `Monitoring (Sampling every ${payload.interval_seconds}s)`;
          }
          if (samplingStatusText) {
            samplingStatusText.textContent = `🟢 Watchdog Active (Sampling ${payload.interval_seconds}s)`;
            samplingStatusText.className = "text-xs font-medium text-emerald-400";
          }
          renderRoiCanvas();
        }
        return;
      }

      if (msg.type === "sample_completed") {
        const payload = msg.data;
        
        // Dynamically adjust interval: 5s during active target detection, 10s baseline
        if (payload.current_interval_seconds) {
          sampleInterval = payload.current_interval_seconds;
        }

        const isHit = payload.detected || payload.rat_detected;
        const meta = getObjectMeta(payload.object_type, payload.label);
        const tagEmoji = meta.emoji;
        const tagLabel = meta.label;

        if (isHit && payload.bounding_box) {
          lastDetectedTargetBox = payload.bounding_box;
          lastDetectedTargetTime = Date.now();
          lastDetectedTargetLabel = tagLabel;
          lastDetectedTargetType = payload.object_type || "target";
          lastDetectedTargetConf = payload.confidence || 0.0;
        } else {
          // Clear bounding box if not an animal
          lastDetectedTargetBox = null;
          renderRoiCanvas();
        }

        // Update header indicator badge
        if (samplingStatusText) {
          if (payload.is_boosted || isHit) {
            samplingStatusText.textContent = `⚡ Tracking ${tagLabel}`;
            samplingStatusText.className = "text-xs font-bold text-amber-400 animate-pulse";
            if (livePulse) livePulse.className = "animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75";
            if (liveDot) liveDot.className = "relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500";
          } else {
            samplingStatusText.textContent = `🟢 Watchdog Active (0% GPU)`;
            samplingStatusText.className = "text-xs font-medium text-emerald-400";
            if (livePulse) livePulse.className = "animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75";
            if (liveDot) liveDot.className = "relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500";
          }
        }

        // Only update image from background sampler if NOT currently receiving real-time live video stream
        const isLiveStreamingNow = (Date.now() - lastLiveVideoFrameTime < 4000);
        if (payload.latest_image_base64 && (!isLiveStreamingNow || isHit)) {
          liveFeedImg.src = payload.latest_image_base64;
          renderRoiCanvas();
        }

        if (!isLiveStreamingNow) {
          liveFrameTime.textContent = payload.timestamp;
          liveFrameAnalysisStatus.textContent = isHit
            ? `${tagEmoji} ${tagLabel} Detected! (${Math.round(payload.confidence * 100)}%)`
            : `Clear (${Math.round(payload.confidence * 100)}%)`;
        }

        // Speak aloud voice alert if enabled
        if (isHit && (!settingVoiceAlert || settingVoiceAlert.checked)) {
          speakAlert(`Notice: ${tagLabel} detected on ${payload.device_health?.device_name || "camera"}.`);
        }

        // Update battery telemetry
        if (payload.device_health) {
          updateBatteryDisplay(
            payload.device_health.battery_percentage,
            payload.device_health.is_mock,
            payload.device_health.device_name,
            payload.device_health.wifi_signal_strength
          );
        }

        if (payload.inference_time_ms) {
          inferenceLatencyLabel.textContent = `${Math.round(payload.inference_time_ms)} ms`;
        }

        if (payload.detection_saved) {
          fetchDetections();
        }
      }
    } catch (e) {
      console.error("WS Parse error:", e);
    }
  };

  socket.onclose = () => {
    setTimeout(initWebSocket, 3000);
  };
}

// --- Backlog Processing Handlers ---
const backlogModal = document.getElementById("backlogModal");
const btnOpenBacklog = document.getElementById("btnOpenBacklog");
const btnCloseBacklog = document.getElementById("btnCloseBacklog");
const btnCloseBacklogBottom = document.getElementById("btnCloseBacklogBottom");
const backlogFileInput = document.getElementById("backlogFileInput");
const btnProcessUploadedBatch = document.getElementById("btnProcessUploadedBatch");
const btnProcessRingHistory = document.getElementById("btnProcessRingHistory");
const ringHistoryLimit = document.getElementById("ringHistoryLimit");
const backlogProgressArea = document.getElementById("backlogProgressArea");
const backlogProgressLabel = document.getElementById("backlogProgressLabel");
const backlogProgressCount = document.getElementById("backlogProgressCount");
const backlogProgressBar = document.getElementById("backlogProgressBar");

if (btnOpenBacklog) {
  btnOpenBacklog.addEventListener("click", () => backlogModal.classList.remove("hidden"));
}
if (btnCloseBacklog) {
  btnCloseBacklog.addEventListener("click", () => backlogModal.classList.add("hidden"));
}
if (btnCloseBacklogBottom) {
  btnCloseBacklogBottom.addEventListener("click", () => {
    backlogModal.classList.add("hidden");
    fetchDetections();
  });
}

if (btnProcessUploadedBatch) {
  btnProcessUploadedBatch.addEventListener("click", async () => {
    const files = backlogFileInput.files;
    if (!files || files.length === 0) {
      alert("Please select one or more image files first.");
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    btnProcessUploadedBatch.disabled = true;
    backlogProgressArea.classList.remove("hidden");
    backlogProgressLabel.textContent = `Analyzing ${files.length} images with Gemma...`;
    backlogProgressCount.textContent = `0 / ${files.length}`;
    backlogProgressBar.style.width = "10%";

    try {
      const res = await fetch("/api/backlog/upload", {
        method: "POST",
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        backlogProgressLabel.textContent = `Done! ${data.positive_detections} rodents detected out of ${data.total_processed} images.`;
        backlogProgressCount.textContent = `${data.total_processed} / ${data.total_processed}`;
        backlogProgressBar.style.width = "100%";
        await fetchDetections();
      }
    } catch (e) {
      console.error("Backlog upload error:", e);
      backlogProgressLabel.textContent = "Error processing batch.";
    } finally {
      btnProcessUploadedBatch.disabled = false;
    }
  });
}

if (btnProcessRingHistory) {
  btnProcessRingHistory.addEventListener("click", async () => {
    const limit = ringHistoryLimit.value;
    btnProcessRingHistory.disabled = true;
    backlogProgressArea.classList.remove("hidden");
    backlogProgressLabel.textContent = `Scanning past ${limit} Ring recordings...`;
    backlogProgressBar.style.width = "25%";

    try {
      const res = await fetch(`/api/backlog/ring_history?limit=${limit}`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        backlogProgressLabel.textContent = `Scan complete! ${data.positive_detections || 0} rodents found in history.`;
        backlogProgressBar.style.width = "100%";
        await fetchDetections();
      }
    } catch (e) {
      console.error("Ring history scan error:", e);
      backlogProgressLabel.textContent = "Error scanning Ring history.";
    } finally {
      btnProcessRingHistory.disabled = false;
    }
  });
}

// --- Voice Alert Player with Smart Deduplication Cooldown ---
let lastSpokenAlertTimes = {};
let browserAlertCooldownMs = 60000; // 60s cooldown for same target/camera

function speakAlert(text = "Warning: Rat detected on camera!", objectType = "rat", force = false) {
  try {
    const key = (objectType || "rat").toLowerCase().trim();
    const now = Date.now();
    const lastTime = lastSpokenAlertTimes[key] || 0;
    
    if (!force && (now - lastTime < browserAlertCooldownMs)) {
      console.log(`[Alert Deduplication] Suppressed repeated voice alert for '${key}' (${Math.round((browserAlertCooldownMs - (now - lastTime)) / 1000)}s remaining in cooldown)`);
      return;
    }
    
    lastSpokenAlertTimes[key] = now;
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.1;
      window.speechSynthesis.speak(utterance);
    }
  } catch (e) {
    console.warn("SpeechSynthesis error:", e);
  }
}

// --- Watch Alert Test Button ---
const btnTestWatchAlert = document.getElementById("btnTestWatchAlert");
const settingEnableNotifs = document.getElementById("settingEnableNotifs");
const settingVoiceAlert = document.getElementById("settingVoiceAlert");
const settingNtfyTopic = document.getElementById("settingNtfyTopic");
const settingTgBotToken = document.getElementById("settingTgBotToken");
const settingTgChatId = document.getElementById("settingTgChatId");
const linkOpenNtfy = document.getElementById("linkOpenNtfy");
const testAlertStatus = document.getElementById("testAlertStatus");

if (settingNtfyTopic && linkOpenNtfy) {
  settingNtfyTopic.addEventListener("input", () => {
    const topic = settingNtfyTopic.value.trim() || "rodentwatch_alerts";
    linkOpenNtfy.href = `https://ntfy.sh/${topic}`;
  });
}

if (btnTestWatchAlert) {
  btnTestWatchAlert.addEventListener("click", async () => {
    btnTestWatchAlert.disabled = true;
    btnTestWatchAlert.textContent = "🔊 Sending WAN Test Alert...";
    if (testAlertStatus) {
      testAlertStatus.classList.remove("hidden");
      testAlertStatus.textContent = "Broadcasting to WAN channels...";
      testAlertStatus.className = "text-[11px] font-mono mt-2 text-center p-2 rounded-lg bg-slate-950 border border-slate-800 text-amber-400";
    }
    speakAlert("Test alert. Rat detected on camera.");

    try {
      const res = await fetch("/api/notifications/test", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        const details = data.result?.details || {};
        let statusMsg = "";
        if (details.ntfy) statusMsg += "✅ ntfy WAN Push: Delivered! ";
        if (details.telegram) statusMsg += "✅ Telegram: Sent! ";
        if (details.ntfy_error) statusMsg += `⚠️ ntfy: ${details.ntfy_error} `;
        if (details.telegram_error) statusMsg += `⚠️ Telegram: ${details.telegram_error} `;

        btnTestWatchAlert.textContent = "✅ Test Broadcast Completed";
        if (testAlertStatus) {
          testAlertStatus.textContent = statusMsg || "✅ Alert sent across WAN channels!";
          testAlertStatus.className = "text-[11px] font-mono mt-2 text-center p-2 rounded-lg bg-slate-950 border border-emerald-800 text-emerald-300";
        }
        setTimeout(() => {
          btnTestWatchAlert.textContent = "🔊 Test Watch & Voice Alert Now";
          btnTestWatchAlert.disabled = false;
        }, 3000);
      }
    } catch (e) {
      console.error("Test notification error:", e);
      btnTestWatchAlert.textContent = "❌ Failed to send alert";
      if (testAlertStatus) {
        testAlertStatus.textContent = `Error: ${e.message}`;
        testAlertStatus.className = "text-[11px] font-mono mt-2 text-center p-2 rounded-lg bg-slate-950 border border-red-800 text-red-300";
      }
      btnTestWatchAlert.disabled = false;
    }
  });
}

// --- Vision Engine Dynamic UI Indicator ---
function updateVisionEngineUi(endpointType, geminiModel, ollamaModel) {
  const isGemini = endpointType === "gemini";
  let engineDisplayName = "Gemma 4 12B";
  let engineShortName = "Gemma 4";
  let engineBadgeClass = "text-xs bg-amber-500/10 text-amber-400 border border-amber-500/30 px-2.5 py-0.5 rounded-full font-medium";
  let cardBadgeClass = "text-[10px] bg-blue-500/10 text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded font-mono";

  if (isGemini) {
    const model = (geminiModel || "").toLowerCase();
    if (model.includes("3.7")) {
      engineDisplayName = "Gemini 3.7 Flash";
      engineShortName = "Gemini 3.7";
    } else if (model.includes("2.5")) {
      engineDisplayName = "Gemini 2.5 Flash";
      engineShortName = "Gemini 2.5";
    } else {
      engineDisplayName = "Gemini Flash";
      engineShortName = "Gemini";
    }
    engineBadgeClass = "text-xs bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 text-purple-300 border border-purple-500/40 px-2.5 py-0.5 rounded-full font-semibold shadow-sm flex items-center gap-1";
    cardBadgeClass = "text-[10px] bg-purple-500/15 text-purple-300 border border-purple-500/30 px-2 py-0.5 rounded font-mono font-bold";
  }

  // 1. Top Navigation Header Badge
  const headerEngineBadge = document.getElementById("headerEngineBadge");
  if (headerEngineBadge) {
    headerEngineBadge.innerHTML = isGemini ? `✨ ${engineDisplayName}` : `${engineDisplayName} Vision`;
    headerEngineBadge.className = engineBadgeClass;
  }

  // 2. Vision Engine Card 3 Badge & Label
  const cardVisionEngineBadge = document.getElementById("cardVisionEngineBadge");
  if (cardVisionEngineBadge) {
    cardVisionEngineBadge.textContent = engineDisplayName;
    cardVisionEngineBadge.className = cardBadgeClass;
  }
  const inferenceEndpointLabel = document.getElementById("inferenceEndpointLabel");
  if (inferenceEndpointLabel) {
    inferenceEndpointLabel.textContent = isGemini ? `Google AI (${engineDisplayName})` : (ollamaModel || "Ollama Local");
  }

  // 3. Live Camera Viewport Status HUDs
  const statusGardenAnalysis = document.getElementById("statusGardenAnalysis");
  if (statusGardenAnalysis) {
    statusGardenAnalysis.textContent = `${engineShortName} Armed & Monitoring`;
  }
  const statusCam1Analysis = document.getElementById("statusCam1Analysis");
  if (statusCam1Analysis) {
    statusCam1Analysis.textContent = `${engineShortName} Ready for Trigger`;
  }

  // 4. Browser Page Title
  document.title = `RodentWatch AI • Ring Camera & ${engineDisplayName}`;

  // 5. Dynamic text mentions
  document.querySelectorAll(".activeEngineName").forEach(el => {
    el.textContent = engineDisplayName;
  });
}

// --- Settings ---
async function loadSettings() {
  try {
    const res = await fetch("/api/settings");
    if (!res.ok) return;
    const data = await res.json();
    settingInterval.value = data.sample_interval_seconds;
    settingEndpoint.value = data.endpoint_url;
    settingModel.value = data.model_name;
    settingThreshold.value = data.confidence_threshold;

    if (data.endpoint_type === "gemini") {
      if (settingEngineProvider) settingEngineProvider.value = data.gemini_model || "gemini-3.7-flash";
      if (containerGeminiKey) containerGeminiKey.classList.remove("hidden");
      if (containerOllamaInputs) containerOllamaInputs.classList.add("hidden");
    } else {
      if (settingEngineProvider) settingEngineProvider.value = "ollama";
      if (containerGeminiKey) containerGeminiKey.classList.add("hidden");
      if (containerOllamaInputs) containerOllamaInputs.classList.remove("hidden");
    }
    if (settingGeminiKey && data.gemini_api_key) settingGeminiKey.value = data.gemini_api_key;

    setTargetObject("rat");

    if (settingNtfyTopic && data.ntfy_topic) {
      settingNtfyTopic.value = data.ntfy_topic;
      if (linkOpenNtfy) linkOpenNtfy.href = `https://ntfy.sh/${data.ntfy_topic}`;
    }
    if (settingTgBotToken && data.telegram_bot_token) settingTgBotToken.value = data.telegram_bot_token;
    if (settingTgChatId && data.telegram_chat_id) settingTgChatId.value = data.telegram_chat_id;
    if (settingPhoneCamUrl && data.phone_camera_url) settingPhoneCamUrl.value = data.phone_camera_url;
    if (settingEnableNotifs && data.enable_notifications !== undefined) settingEnableNotifs.checked = data.enable_notifications;
    if (settingVoiceAlert && data.voice_alert !== undefined) settingVoiceAlert.checked = data.voice_alert;

    // Load per-camera ROI polygons
    if (data.camera_polygons && typeof data.camera_polygons === "object") {
      Object.keys(data.camera_polygons).forEach(cam => {
        const poly = data.camera_polygons[cam];
        if (Array.isArray(poly) && poly.length >= 3) {
          const cLow = cam.toLowerCase();
          const key = cLow.includes("garden") ? "Garden" : (cLow.includes("cam1") ? "cam1" : "S21");
          cameraPolygons[key] = poly.map(pt => ({ x: pt[0], y: pt[1] }));
        }
      });
    }

    if (data.active_camera) {
      const aLow = data.active_camera.toLowerCase();
      currentActiveCamera = aLow.includes("cam1") ? "cam1" : (aLow.includes("s21") || aLow.includes("s1") || aLow.includes("phone") ? "S21" : "Garden");
    }

    if (data.detection_polygon && Array.isArray(data.detection_polygon) && data.detection_polygon.length >= 3) {
      activePolygon = data.detection_polygon.map(pt => ({ x: pt[0], y: pt[1] }));
      if (!cameraPolygons[currentActiveCamera]) {
        cameraPolygons[currentActiveCamera] = activePolygon;
      }
    }

    updateRoiUiState();
    renderRoiCanvas();

    // Dynamically update all UI indicators to match active vision engine
    updateVisionEngineUi(data.endpoint_type, data.gemini_model, data.model_name);

    inferenceThresholdLabel.textContent = `≥ ${Math.round(data.confidence_threshold * 100)}%`;
    sampleInterval = data.sample_interval_seconds;
  } catch (err) {
    console.error("Error loading settings:", err);
  }
}

async function saveSettings() {
  const isGemini = settingEngineProvider && settingEngineProvider.value.startsWith("gemini");
  const payload = {
    sample_interval_seconds: parseInt(settingInterval.value, 10),
    endpoint_type: isGemini ? "gemini" : "ollama",
    endpoint_url: settingEndpoint.value.trim(),
    model_name: isGemini ? settingEngineProvider.value : settingModel.value.trim(),
    gemini_model: isGemini ? settingEngineProvider.value : "gemini-3.7-flash",
    gemini_api_key: settingGeminiKey ? settingGeminiKey.value.trim() : null,
    confidence_threshold: parseFloat(settingThreshold.value),
    target_object: "rat",
    ntfy_topic: settingNtfyTopic ? settingNtfyTopic.value.trim() : "rodentwatch_alerts",
    enable_notifications: settingEnableNotifs ? settingEnableNotifs.checked : true,
    voice_alert: settingVoiceAlert ? settingVoiceAlert.checked : true,
    telegram_bot_token: settingTgBotToken ? settingTgBotToken.value.trim() : null,
    telegram_chat_id: settingTgChatId ? settingTgChatId.value.trim() : null,
    phone_camera_url: settingPhoneCamUrl ? settingPhoneCamUrl.value.trim() : null
  };

  try {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      sampleInterval = data.current_settings.sample_interval_seconds;
      updateVisionEngineUi(
        data.current_settings.endpoint_type,
        data.current_settings.gemini_model,
        data.current_settings.model_name
      );
      inferenceThresholdLabel.textContent = `≥ ${Math.round(data.current_settings.confidence_threshold * 100)}%`;
      if (data.current_settings.target_object) {
        setTargetObject(data.current_settings.target_object);
      }
      settingsModal.classList.add("hidden");
      nextSampleSeconds = sampleInterval;
    }
  } catch (err) {
    console.error("Error saving settings:", err);
  }
}

// --- Interactive ROI Polygon Drawing System (Independent Per-Camera) ---
function getRoiCanvases() {
  const list = [];
  if (mainCameraRoiCanvas) list.push(mainCameraRoiCanvas);
  if (roiCanvas && !list.includes(roiCanvas)) list.push(roiCanvas);
  return list;
}

function getCameraForCanvas(canvas) {
  return currentSelectedCamera || "Garden";
}

function getPointAtCoordinates(normX, normY, poly, rect) {
  if (!poly || !Array.isArray(poly) || poly.length === 0) return -1;
  const thresholdPx = 16;
  for (let i = 0; i < poly.length; i++) {
    const pt = poly[i];
    const dist = Math.hypot((normX - pt.x) * rect.width, (normY - pt.y) * rect.height);
    if (dist <= thresholdPx) return i;
  }
  return -1;
}

function initRoiDrawing() {
  try {
    const cached = JSON.parse(localStorage.getItem("rodent_camera_polygons") || "{}");
    if (cached.Garden && Array.isArray(cached.Garden) && cached.Garden.length >= 3) {
      cameraPolygons.Garden = cached.Garden;
    }
    if (cached.cam1 && Array.isArray(cached.cam1) && cached.cam1.length >= 3) {
      cameraPolygons.cam1 = cached.cam1;
    }
    if (cached.S21 && Array.isArray(cached.S21) && cached.S21.length >= 3) {
      cameraPolygons.S21 = cached.S21;
    }
    const legacy = localStorage.getItem("rodent_detection_polygon");
    if (legacy && !cameraPolygons[currentActiveCamera]) {
      const parsed = JSON.parse(legacy);
      if (Array.isArray(parsed) && parsed.length >= 3) {
        cameraPolygons[currentActiveCamera] = parsed;
        activePolygon = parsed;
      }
    }
  } catch (e) {}

  const canvases = getRoiCanvases();
  canvases.forEach((cvs) => {
    cvs.addEventListener("mousedown", handleCanvasMouseDown);
    cvs.addEventListener("click", handleCanvasClick);
    cvs.addEventListener("mousemove", handleCanvasMouseMove);
    cvs.addEventListener("mouseleave", handleCanvasMouseLeave);
    cvs.addEventListener("dblclick", handleCanvasDblClick);
  });
  window.addEventListener("mouseup", handleCanvasMouseUp);

  const resizeObserver = new ResizeObserver(() => {
    resizeRoiCanvas();
  });
  if (mainSurveillanceContainer) resizeObserver.observe(mainSurveillanceContainer);
  if (surveillanceTransformLayer) resizeObserver.observe(surveillanceTransformLayer);
  window.addEventListener("resize", resizeRoiCanvas);
  if (mainCameraFeedImg) mainCameraFeedImg.addEventListener("load", resizeRoiCanvas);
  setTimeout(resizeRoiCanvas, 300);

  if (btnDrawActiveZone) {
    btnDrawActiveZone.addEventListener("click", () => toggleDrawZoneForCamera(currentSelectedCamera));
  }
  if (btnSaveActiveZone) {
    btnSaveActiveZone.addEventListener("click", saveActiveCameraZone);
  }
  if (btnClearActiveZone) {
    btnClearActiveZone.addEventListener("click", () => clearZoneFor(currentSelectedCamera));
  }

  // Esc key cancels active drawing mode
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isDrawingZone) {
      cancelZoneDrawing();
    }
  });

  initStreamWatchdog();
  updateRoiUiState();
  renderRoiCanvas();
}

function resizeRoiCanvas() {
  const canvases = getRoiCanvases();
  canvases.forEach((cvs) => {
    const parent = cvs.parentElement;
    if (parent) {
      const rect = parent.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        cvs.width = rect.width;
        cvs.height = rect.height;
      }
    }
  });
  renderRoiCanvas();
}

function toggleDrawZoneForCamera(cameraName) {
  if (isDrawingZone && currentDrawingCamera === cameraName) {
    if (currentPolygon.length >= 3) {
      finishZoneDrawing();
    } else {
      cancelZoneDrawing();
    }
  } else {
    currentDrawingCamera = cameraName;
    isDrawingZone = true;

    // If existing polygon exists, load into currentPolygon for dragging and editing
    const existing = cameraPolygons[cameraName] || (cameraName === currentActiveCamera ? activePolygon : null);
    if (existing && Array.isArray(existing) && existing.length >= 3) {
      currentPolygon = existing.map(pt => ({ ...pt }));
    } else {
      currentPolygon = [];
    }
    mousePos = null;

    const canvases = getRoiCanvases();
    canvases.forEach((cvs) => {
      cvs.classList.remove("pointer-events-none");
      cvs.classList.add("pointer-events-auto");
      cvs.style.cursor = "crosshair";
    });

    if (drawInstructions) {
      drawInstructions.innerHTML = `<span>🎯</span> Drag points to reshape or click to add points on <b>${cameraName}</b>. Double-click or click start point to finish.`;
      drawInstructions.classList.remove("hidden");
    }

    if (btnDrawActiveZone && textDrawActiveZone) {
      btnDrawActiveZone.className = "px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-bold transition flex items-center gap-1.5 shadow-lg animate-pulse active:scale-95";
      textDrawActiveZone.textContent = "Finish Zone";
    }

    renderRoiCanvas();
  }
}

function cancelZoneDrawing() {
  isDrawingZone = false;
  currentPolygon = [];
  mousePos = null;
  activeDrawCanvas = null;
  hoveredPointIndex = -1;
  isDraggingPoint = false;
  draggedPointIndex = -1;

  if (drawInstructions) drawInstructions.classList.add("hidden");
  updateRoiUiState();
  renderRoiCanvas();
}

function handleCanvasMouseDown(e) {
  const canvas = e.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const normX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const normY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

  const camName = currentSelectedCamera;
  const activePoly = (isDrawingZone && currentDrawingCamera === camName && currentPolygon.length > 0)
    ? currentPolygon
    : (cameraPolygons[camName] || (camName === currentActiveCamera ? activePolygon : null));

  const ptIdx = getPointAtCoordinates(normX, normY, activePoly, rect);
  if (ptIdx >= 0) {
    isDraggingPoint = true;
    draggedPointIndex = ptIdx;
    hasJustDragged = false;
    canvas.style.cursor = "grabbing";
    e.preventDefault();
    e.stopPropagation();
  }
}

function handleCanvasMouseMove(e) {
  const canvas = e.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const normX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const normY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

  const camName = currentSelectedCamera;
  const activePoly = (isDrawingZone && currentDrawingCamera === camName && currentPolygon.length > 0)
    ? currentPolygon
    : (cameraPolygons[camName] || (camName === currentActiveCamera ? activePolygon : null));

  // 1. Dragging an existing vertex point
  if (isDraggingPoint && draggedPointIndex >= 0 && activePoly && activePoly[draggedPointIndex]) {
    hasJustDragged = true;
    activePoly[draggedPointIndex] = {
      x: Number(normX.toFixed(4)),
      y: Number(normY.toFixed(4))
    };
    renderRoiCanvas();
    return;
  }

  // 2. Hover detection over vertex points
  const ptIdx = getPointAtCoordinates(normX, normY, activePoly, rect);
  if (ptIdx !== hoveredPointIndex) {
    hoveredPointIndex = ptIdx;
    renderRoiCanvas();
  }

  if (ptIdx >= 0) {
    canvas.style.cursor = "grab";
  } else if (isDrawingZone) {
    canvas.style.cursor = "crosshair";
    mousePos = { x: normX, y: normY, canvas };
    renderRoiCanvas();
  } else {
    canvas.style.cursor = "default";
    if (mousePos) {
      mousePos = null;
      renderRoiCanvas();
    }
  }
}

async function handleCanvasMouseUp(e) {
  if (isDraggingPoint) {
    isDraggingPoint = false;
    draggedPointIndex = -1;
    const canvas = mainCameraRoiCanvas;
    if (canvas) {
      canvas.style.cursor = (hoveredPointIndex >= 0) ? "grab" : "default";
    }

    const cam = currentSelectedCamera;
    const poly = (isDrawingZone && currentDrawingCamera === cam && currentPolygon.length > 0)
      ? currentPolygon
      : cameraPolygons[cam];

    if (poly && poly.length >= 3) {
      cameraPolygons[cam] = [...poly];
      if (cam === currentActiveCamera) {
        activePolygon = [...poly];
      }
      try {
        const cached = JSON.parse(localStorage.getItem("rodent_camera_polygons") || "{}");
        cached[cam] = cameraPolygons[cam];
        localStorage.setItem("rodent_camera_polygons", JSON.stringify(cached));
        if (cam === currentActiveCamera) {
          localStorage.setItem("rodent_detection_polygon", JSON.stringify(cameraPolygons[cam]));
        }
      } catch (err) {}

      // Debounced sync to backend
      try {
        const polyArray = cameraPolygons[cam].map(p => [Number(p.x.toFixed(4)), Number(p.y.toFixed(4))]);
        await fetch(`/api/camera/${cam}/zone`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ polygon: polyArray })
        });
      } catch (err) {}
    }

    renderRoiCanvas();
    updateRoiUiState();
    setTimeout(() => { hasJustDragged = false; }, 80);
  }
}

function handleCanvasMouseLeave(e) {
  if (hoveredPointIndex >= 0 && !isDraggingPoint) {
    hoveredPointIndex = -1;
    renderRoiCanvas();
  }
  if (mousePos && !isDraggingPoint) {
    mousePos = null;
    renderRoiCanvas();
  }
}

function handleCanvasClick(e) {
  if (hasJustDragged) {
    hasJustDragged = false;
    return;
  }
  if (!isDrawingZone) return;

  const canvas = e.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const clickNormX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const clickNormY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

  const pt = {
    x: Number(clickNormX.toFixed(4)),
    y: Number(clickNormY.toFixed(4))
  };

  // If clicking near first point (when >= 3 points), close polygon
  if (currentPolygon.length >= 3) {
    const first = currentPolygon[0];
    const distPx = Math.hypot((clickNormX - first.x) * rect.width, (clickNormY - first.y) * rect.height);
    if (distPx < 22) {
      finishZoneDrawing();
      return;
    }
  }

  currentPolygon.push(pt);
  renderRoiCanvas();
}

function handleCanvasDblClick(e) {
  if (isDrawingZone && currentPolygon.length >= 3) {
    e.preventDefault();
    finishZoneDrawing();
  }
}

async function finishZoneDrawing() {
  if (currentPolygon.length < 3) {
    cancelZoneDrawing();
    return;
  }

  const cam = currentDrawingCamera || currentSelectedCamera || "Garden";
  cameraPolygons[cam] = [...currentPolygon];

  if (cam.toLowerCase() === (currentActiveCamera || "").toLowerCase()) {
    activePolygon = [...currentPolygon];
  }

  isDrawingZone = false;
  currentPolygon = [];
  mousePos = null;
  activeDrawCanvas = null;
  hoveredPointIndex = -1;
  isDraggingPoint = false;
  draggedPointIndex = -1;

  if (drawInstructions) drawInstructions.classList.add("hidden");

  try {
    const cached = JSON.parse(localStorage.getItem("rodent_camera_polygons") || "{}");
    cached[cam] = cameraPolygons[cam];
    localStorage.setItem("rodent_camera_polygons", JSON.stringify(cached));
    if (cam === currentActiveCamera) {
      localStorage.setItem("rodent_detection_polygon", JSON.stringify(cameraPolygons[cam]));
    }
  } catch (e) {}

  updateRoiUiState();
  renderRoiCanvas();

  // Save polygon to server for this specific camera
  try {
    const polyArray = cameraPolygons[cam].map(p => [Number(p.x.toFixed(4)), Number(p.y.toFixed(4))]);
    await fetch(`/api/camera/${cam}/zone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: polyArray })
    });
  } catch (err) {
    console.error(`Error saving detection polygon for ${cam}:`, err);
  }
}

async function saveActiveCameraZone() {
  const cam = currentSelectedCamera;
  const poly = cameraPolygons[cam];
  if (!poly || poly.length < 3) {
    if (textActiveCameraZone) {
      textActiveCameraZone.textContent = "Draw 3+ pts first";
      setTimeout(updateRoiUiState, 2000);
    }
    return;
  }
  try {
    const polyArray = poly.map(p => [Number(p.x.toFixed(4)), Number(p.y.toFixed(4))]);
    const res = await fetch(`/api/camera/${cam}/zone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: polyArray })
    });
    if (res.ok) {
      if (textActiveCameraZone) {
        textActiveCameraZone.textContent = `✅ Saved (${poly.length} pts)`;
        setTimeout(updateRoiUiState, 2000);
      }
    }
  } catch (err) {
    console.error(`Error saving zone for ${cam}:`, err);
  }
}

async function clearZoneFor(cameraName) {
  cameraPolygons[cameraName] = null;
  if (cameraName.toLowerCase() === (currentActiveCamera || "").toLowerCase()) {
    activePolygon = [];
  }
  try {
    const cached = JSON.parse(localStorage.getItem("rodent_camera_polygons") || "{}");
    cached[cameraName] = null;
    localStorage.setItem("rodent_camera_polygons", JSON.stringify(cached));
    if (cameraName.toLowerCase() === (currentActiveCamera || "").toLowerCase()) {
      localStorage.removeItem("rodent_detection_polygon");
    }
  } catch (e) {}

  updateRoiUiState();
  renderRoiCanvas();

  try {
    await fetch(`/api/camera/${cameraName}/zone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon: null })
    });
  } catch (err) {
    console.error(`Error clearing zone for ${cameraName}:`, err);
  }
}

function updateRoiUiState() {
  const cam = currentSelectedCamera;
  const camPoly = cameraPolygons[cam] || (cam === currentActiveCamera ? activePolygon : null);
  const hasZone = camPoly && camPoly.length >= 3;

  const canvases = getRoiCanvases();
  canvases.forEach((cvs) => {
    if (isDrawingZone || hasZone) {
      cvs.classList.remove("pointer-events-none");
      cvs.classList.add("pointer-events-auto");
    } else {
      cvs.classList.add("pointer-events-none");
      cvs.classList.remove("pointer-events-auto");
    }
  });

  if (badgeActiveCameraZone && textActiveCameraZone) {
    if (hasZone) {
      badgeActiveCameraZone.className = "text-xs bg-amber-500/20 text-amber-300 border border-amber-500/40 px-3 py-1.5 rounded-xl font-mono font-bold flex items-center gap-1.5 shadow";
      textActiveCameraZone.textContent = `🎯 Zone Active (${camPoly.length} pts)`;
    } else {
      badgeActiveCameraZone.className = "text-xs bg-slate-800 text-slate-400 border border-slate-700 px-3 py-1.5 rounded-xl font-mono font-medium flex items-center gap-1.5";
      textActiveCameraZone.textContent = "⚪ No Zone Set";
    }
  }

  if (btnDrawActiveZone && textDrawActiveZone) {
    if (isDrawingZone) {
      btnDrawActiveZone.className = "px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-bold transition flex items-center gap-1.5 shadow-lg animate-pulse active:scale-95";
      textDrawActiveZone.textContent = "Finish Zone";
    } else {
      btnDrawActiveZone.className = "px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-xs font-bold transition flex items-center gap-1.5 shadow active:scale-95";
      textDrawActiveZone.textContent = hasZone ? "Edit Zone" : "Draw Zone";
    }
  }

  if (btnClearActiveZone) {
    btnClearActiveZone.classList.toggle("opacity-50", !hasZone);
  }
}

function renderRoiCanvas() {
  const canvas = mainCameraRoiCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  if (w <= 0 || h <= 0) return;

  ctx.clearRect(0, 0, w, h);

  const camName = currentSelectedCamera;
  const camPoly = cameraPolygons[camName] || (camName === currentActiveCamera ? activePolygon : null);

  // 1. Render Saved Polygon for current camera
  if (camPoly && camPoly.length >= 3 && (!isDrawingZone || currentDrawingCamera !== camName)) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(camPoly[0].x * w, camPoly[0].y * h);
    for (let i = 1; i < camPoly.length; i++) {
      ctx.lineTo(camPoly[i].x * w, camPoly[i].y * h);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(245, 158, 11, 0.18)";
    ctx.fill();

    // Thinner, crisp stroke with subtle glow
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.25;
    ctx.shadowColor = "rgba(245, 158, 11, 0.4)";
    ctx.shadowBlur = 2;
    ctx.stroke();

    ctx.shadowBlur = 0;
    for (let i = 0; i < camPoly.length; i++) {
      const px = camPoly[i].x * w;
      const py = camPoly[i].y * h;
      const isHovered = (hoveredPointIndex === i);
      const isDragged = (isDraggingPoint && draggedPointIndex === i);

      ctx.beginPath();
      const r = (isDragged || isHovered) ? 6 : 3.5;
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = isDragged ? "#ffffff" : isHovered ? "#fef08a" : "#fbbf24";
      ctx.fill();
      ctx.strokeStyle = isDragged ? "#b45309" : "#1e293b";
      ctx.lineWidth = 1.0;
      ctx.stroke();

      // Pulsing outer halo when hovered or being dragged
      if (isHovered || isDragged) {
        ctx.beginPath();
        ctx.arc(px, py, r + 3.5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(245, 158, 11, 0.75)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    // Zone Badge Tag inside polygon
    const centerNormX = camPoly.reduce((sum, p) => sum + p.x, 0) / camPoly.length;
    const centerNormY = camPoly.reduce((sum, p) => sum + p.y, 0) / camPoly.length;
    const centerX = centerNormX * w;
    const centerY = centerNormY * h;

    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
    ctx.fillRect(centerX - 50, centerY - 11, 100, 22);
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1;
    ctx.strokeRect(centerX - 50, centerY - 11, 100, 22);
    ctx.fillStyle = "#fbbf24";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`🎯 ${camName} Zone`, centerX, centerY);

    ctx.restore();
  }

  // 2. Render In-Progress Drawing Polygon
  if (isDrawingZone && currentDrawingCamera === camName && currentPolygon.length > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(currentPolygon[0].x * w, currentPolygon[0].y * h);
    for (let i = 1; i < currentPolygon.length; i++) {
      ctx.lineTo(currentPolygon[i].x * w, currentPolygon[i].y * h);
    }
    if (mousePos) {
      ctx.lineTo(mousePos.x * w, mousePos.y * h);
    }
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 1.25;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    if (currentPolygon.length >= 3) {
      ctx.fillStyle = "rgba(16, 185, 129, 0.15)";
      ctx.fill();
    }

    for (let i = 0; i < currentPolygon.length; i++) {
      const px = currentPolygon[i].x * w;
      const py = currentPolygon[i].y * h;
      const isHovered = (hoveredPointIndex === i);
      const isDragged = (isDraggingPoint && draggedPointIndex === i);

      ctx.beginPath();
      const r = (isDragged || isHovered) ? 6 : (i === 0 ? 5 : 3.5);
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = isDragged ? "#ffffff" : isHovered ? "#fef08a" : (i === 0 ? "#ef4444" : "#10b981");
      ctx.fill();
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1.0;
      ctx.stroke();

      if (i === 0) {
        ctx.font = "10px Inter, sans-serif";
        ctx.fillStyle = "#ef4444";
        ctx.fillText("Start", px + 8, py - 6);
      }
    }
    ctx.restore();
  }

  // 3. Render Detected Target Bounding Box
  if (lastDetectedTargetBox && (Date.now() - lastDetectedTargetTime < 15000)) {
    renderDetectedTargetBoundingBox(ctx, w, h, lastDetectedTargetBox, lastDetectedTargetLabel, lastDetectedTargetType, lastDetectedTargetConf);
  }
}
window.cameraPolygons = cameraPolygons;
window.renderRoiCanvas = renderRoiCanvas;
window.updateRoiUiState = updateRoiUiState;
window.setDetectedTarget = (box, label, type, conf) => {
  lastDetectedTargetBox = box;
  lastDetectedTargetTime = Date.now();
  lastDetectedTargetLabel = label;
  lastDetectedTargetType = type;
  lastDetectedTargetConf = conf;
  renderRoiCanvas();
};

// =========================================================================
// --- REAL-TIME SCREEN CAM, DIGITAL ZOOM & MOTION DETECTION SYSTEM ---
// =========================================================================

// Stream Source DOM Elements
const btnToggleWebcam = document.getElementById("btnToggleWebcam");
const webcamIcon = document.getElementById("webcamIcon");
const webcamText = document.getElementById("webcamText");
const btnToggleScreenCam = document.getElementById("btnToggleScreenCam");
const screenCamIcon = document.getElementById("screenCamIcon");
const screenCamText = document.getElementById("screenCamText");
const screenCamStatus = document.getElementById("screenCamStatus");
const liveStreamSourceLabel = document.getElementById("liveStreamSourceLabel");
const screenCamMotionBadge = document.getElementById("screenCamMotionBadge");
const btnToggleClarityBoost = document.getElementById("btnToggleClarityBoost");
const clarityBoostIcon = document.getElementById("clarityBoostIcon");
const clarityBoostText = document.getElementById("clarityBoostText");

// Digital Zoom DOM Elements
const zoomSlider = document.getElementById("zoomSlider");
const btnZoomIn = document.getElementById("btnZoomIn");
const btnZoomOut = document.getElementById("btnZoomOut");
const btnResetZoom = document.getElementById("btnResetZoom");
const zoomLevelLabel = document.getElementById("zoomLevelLabel");
const quickZoomButtons = document.querySelectorAll(".btn-quick-zoom");

// Stream State (Webcam & Screen Cam)
let activeLiveStream = null;
let activeLiveVideo = null;
let isLiveStreaming = false;
let liveStreamType = null; // 'webcam' or 'screen'
let isClarityBoostActive = false;
let screenCamAnimId = null;
let currentVerticalStretch = 1.75; // Default 1.75x vertical un-squash stretch

function setVerticalStretch(factor) {
  currentVerticalStretch = factor;
  if (liveFeedImg) {
    liveFeedImg.style.transform = `scaleY(${factor})`;
    liveFeedImg.style.transformOrigin = "center";
  }
  document.querySelectorAll(".btn-stretch-preset").forEach(btn => {
    const s = parseFloat(btn.dataset.stretch);
    if (Math.abs(s - factor) < 0.05) {
      btn.className = "btn-stretch-preset px-2 py-0.5 rounded-lg text-xs font-bold text-amber-400 bg-slate-800 transition";
    } else {
      btn.className = "btn-stretch-preset px-2 py-0.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition";
    }
  });
  renderRoiCanvas();
}

// Digital Zoom & Pan State
let currentZoom = 1.0;
let panX = 0.5; // Normalized center X (0.0 to 1.0)
let panY = 0.5; // Normalized center Y (0.0 to 1.0)
let isDraggingPan = false;
let lastPanMouseX = 0;
let lastPanMouseY = 0;

// Motion Detection State
let motionCanvas = document.createElement("canvas");
let motionCtx = motionCanvas.getContext("2d", { willReadFrequently: true });
let prevMotionData = null;
let lastMotionCheckTime = 0;
let isGemmaAnalyzingScreen = false;
let lastGemmaTriggerTime = 0;
let lastDetectedTargetBox = null;
let lastDetectedTargetTime = 0;
let lastDetectedTargetLabel = "Rat";
let lastDetectedTargetType = "rat";
let lastDetectedTargetConf = 0.0;
let lastDetectedRatBox = null;
let lastDetectedRatTime = 0;

function initScreenCamAndZoom() {
  if (btnToggleWebcam) {
    btnToggleWebcam.addEventListener("click", toggleWebcam);
  }
  if (btnToggleScreenCam) {
    btnToggleScreenCam.addEventListener("click", toggleScreenCam);
  }

  // Camera selector dropdown auto-trigger
  if (cameraSelectDropdown) {
    cameraSelectDropdown.addEventListener("change", (e) => {
      const val = e.target.value;
      if (val.includes("Local Camera") && !isLiveStreaming) {
        startWebcam();
      }
    });
  }

  // Zoom slider & buttons
  if (zoomSlider) {
    zoomSlider.addEventListener("input", (e) => {
      setZoomLevel(parseFloat(e.target.value));
    });
  }
  if (btnZoomIn) {
    btnZoomIn.addEventListener("click", () => setZoomLevel(Math.min(5.0, currentZoom + 0.5)));
  }
  if (btnZoomOut) {
    btnZoomOut.addEventListener("click", () => setZoomLevel(Math.max(1.0, currentZoom - 0.5)));
  }
  if (btnResetZoom) {
    btnResetZoom.addEventListener("click", () => {
      panX = 0.5;
      panY = 0.5;
      setZoomLevel(1.0);
    });
  }

  quickZoomButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const z = parseFloat(btn.getAttribute("data-zoom") || "1.0");
      setZoomLevel(z);
    });
  });

  if (btnToggleClarityBoost) {
    btnToggleClarityBoost.addEventListener("click", () => {
      isClarityBoostActive = !isClarityBoostActive;
      if (isClarityBoostActive) {
        btnToggleClarityBoost.className = "ml-1 px-2.5 py-1 bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold rounded-lg transition flex items-center gap-1 active:scale-95 shadow-md shadow-amber-600/30 animate-pulse";
        clarityBoostIcon.textContent = "☀️";
        clarityBoostText.textContent = "Boost: ON";
      } else {
        btnToggleClarityBoost.className = "ml-1 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-amber-300 text-xs font-semibold rounded-lg transition flex items-center gap-1 active:scale-95";
        clarityBoostIcon.textContent = "🌙";
        clarityBoostText.textContent = "Night Boost";
      }
      if (!isLiveStreaming && liveFeedImg) {
        liveFeedImg.style.filter = isClarityBoostActive ? "contrast(1.3) brightness(1.15) saturate(1.2)" : "none";
      }
    });
  }

  // Mouse wheel zoom inside live feed container
  if (liveFeedContainer) {
    liveFeedContainer.addEventListener("wheel", (e) => {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.2 : -0.2;
      setZoomLevel(Math.max(1.0, Math.min(5.0, currentZoom + delta)));
    }, { passive: false });

    // Mouse drag to pan when zoomed in (and not drawing ROI zone)
    liveFeedContainer.addEventListener("mousedown", (e) => {
      if (isDrawingZone || currentZoom <= 1.05) return;
      if (e.button === 0) { // Left click
        isDraggingPan = true;
        lastPanMouseX = e.clientX;
        lastPanMouseY = e.clientY;
        liveFeedContainer.style.cursor = "grabbing";
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDraggingPan || currentZoom <= 1.05) return;
      const dx = (e.clientX - lastPanMouseX) / (liveFeedContainer.clientWidth * currentZoom);
      const dy = (e.clientY - lastPanMouseY) / (liveFeedContainer.clientHeight * currentZoom);
      panX = Math.max(0.1, Math.min(0.9, panX - dx));
      panY = Math.max(0.1, Math.min(0.9, panY - dy));
      lastPanMouseX = e.clientX;
      lastPanMouseY = e.clientY;
      if (!isScreenCamActive) renderRoiCanvas();
    });

    window.addEventListener("mouseup", () => {
      if (isDraggingPan) {
        isDraggingPan = false;
        liveFeedContainer.style.cursor = "";
      }
    });
  }
}

function setZoomLevel(zoom) {
  currentZoom = Math.round(zoom * 10) / 10;
  if (currentZoom <= 1.0) {
    currentZoom = 1.0;
    panX = 0.5;
    panY = 0.5;
  }
  if (zoomSlider) zoomSlider.value = currentZoom;
  if (zoomLevelLabel) zoomLevelLabel.textContent = `${currentZoom.toFixed(1)}x`;

  // Apply CSS transform to static live feed image when Screen Cam is inactive
  if (!isScreenCamActive && liveFeedImg) {
    if (currentZoom > 1.0) {
      const originX = panX * 100;
      const originY = panY * 100;
      liveFeedImg.style.transformOrigin = `${originX}% ${originY}%`;
      liveFeedImg.style.transform = `scale(${currentZoom})`;
    } else {
      liveFeedImg.style.transform = "none";
    }
  }

  renderRoiCanvas();
}

async function toggleWebcam() {
  if (isLiveStreaming && liveStreamType === "webcam") {
    stopLiveStream();
  } else {
    if (isLiveStreaming) stopLiveStream();
    await startWebcam();
  }
}

let isContinuousMjpeg = true;

const btnToggleLiveVideoMode = document.getElementById("btnToggleLiveVideoMode");
const liveVideoModeText = document.getElementById("liveVideoModeText");

if (btnToggleLiveVideoMode) {
  btnToggleLiveVideoMode.addEventListener("click", toggleLiveVideoMode);
}

function toggleLiveVideoMode() {
  isContinuousMjpeg = !isContinuousMjpeg;
  if (isContinuousMjpeg) {
    liveFeedImg.src = "/api/camera/live_stream?t=" + Date.now();
    btnToggleLiveVideoMode.className = "px-3.5 py-1.5 bg-red-600 hover:bg-red-500 text-white text-xs font-bold rounded-xl transition flex items-center gap-1.5 shadow-lg shadow-red-600/20 active:scale-95";
    liveVideoModeText.textContent = "Live Video Active 🔴";
    if (liveFrameTime) liveFrameTime.textContent = "Live Continuous Stream (30fps 🔴)";
  } else {
    liveFeedImg.src = "/api/camera/latest_snapshot?t=" + Date.now();
    btnToggleLiveVideoMode.className = "px-3.5 py-1.5 bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold rounded-xl transition flex items-center gap-1.5 shadow-lg shadow-amber-600/20 active:scale-95";
    liveVideoModeText.textContent = "Continuous Live Video";
  }
}

// Auto-start continuous live video stream on load
if (liveFeedImg) {
  liveFeedImg.src = "/api/camera/live_stream?t=" + Date.now();
  if (btnToggleLiveVideoMode) {
    btnToggleLiveVideoMode.className = "px-3.5 py-1.5 bg-red-600 hover:bg-red-500 text-white text-xs font-bold rounded-xl transition flex items-center gap-1.5 shadow-lg shadow-red-600/20 active:scale-95";
    if (liveVideoModeText) liveVideoModeText.textContent = "Live Video Active 🔴";
  }
}

async function toggleScreenCam() {
  if (isLiveStreaming && liveStreamType === "screen") {
    stopLiveStream();
  } else {
    if (isLiveStreaming) stopLiveStream();
    await startScreenCam();
  }
}

async function startWebcam() {
  try {
    activeLiveStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30, max: 30 }
      },
      audio: false
    });

    activeLiveVideo = document.createElement("video");
    activeLiveVideo.srcObject = activeLiveStream;
    activeLiveVideo.autoplay = true;
    activeLiveVideo.playsInline = true;
    activeLiveVideo.muted = true;
    await activeLiveVideo.play();

    isLiveStreaming = true;
    liveStreamType = "webcam";

    if (liveFeedImg) liveFeedImg.classList.add("opacity-0");
    if (screenCamStatus) screenCamStatus.classList.remove("hidden");
    if (liveStreamSourceLabel) liveStreamSourceLabel.textContent = "Live Webcam (30fps)";
    if (screenCamMotionBadge) screenCamMotionBadge.classList.remove("hidden");

    if (btnToggleWebcam) {
      btnToggleWebcam.className = "px-3.5 py-1.5 bg-red-600 hover:bg-red-500 text-white text-xs font-bold rounded-xl transition flex items-center gap-1.5 shadow-lg shadow-red-600/20 active:scale-95";
      webcamIcon.textContent = "⏹️";
      webcamText.textContent = "Stop Webcam";
    }

    if (liveFrameTime) liveFrameTime.textContent = "Real-Time (Webcam 30fps)";
    if (liveFrameAnalysisStatus) liveFrameAnalysisStatus.textContent = "⚡ Real-time motion detection active";

    activeLiveStream.getVideoTracks()[0].onended = () => {
      stopLiveStream();
    };

    // Start 30fps animation & motion detection loop
    screenCamLoop();
  } catch (err) {
    console.warn("Webcam access error:", err);
    alert("Could not access Webcam: Please allow camera permissions in your browser.");
  }
}

async function startScreenCam() {
  try {
    activeLiveStream = await navigator.mediaDevices.getDisplayMedia({
      video: {
        frameRate: { ideal: 30, max: 30 },
        cursor: "always"
      },
      audio: false
    });

    activeLiveVideo = document.createElement("video");
    activeLiveVideo.srcObject = activeLiveStream;
    activeLiveVideo.autoplay = true;
    activeLiveVideo.playsInline = true;
    activeLiveVideo.muted = true;
    await activeLiveVideo.play();

    isLiveStreaming = true;
    liveStreamType = "screen";

    if (liveFeedImg) liveFeedImg.classList.add("opacity-0");
    if (screenCamStatus) screenCamStatus.classList.remove("hidden");
    if (liveStreamSourceLabel) liveStreamSourceLabel.textContent = "Live Screen (30fps)";
    if (screenCamMotionBadge) screenCamMotionBadge.classList.remove("hidden");

    if (btnToggleScreenCam) {
      btnToggleScreenCam.className = "px-3.5 py-1.5 bg-red-600 hover:bg-red-500 text-white text-xs font-bold rounded-xl transition flex items-center gap-1.5 shadow-lg shadow-red-600/20 active:scale-95";
      screenCamIcon.textContent = "⏹️";
      screenCamText.textContent = "Stop Screen Cam";
    }

    if (liveFrameTime) liveFrameTime.textContent = "Real-Time (Screen 30fps)";
    if (liveFrameAnalysisStatus) liveFrameAnalysisStatus.textContent = "⚡ Real-time motion detection active";

    activeLiveStream.getVideoTracks()[0].onended = () => {
      stopLiveStream();
    };

    // Start 30fps animation & motion detection loop
    screenCamLoop();
  } catch (err) {
    console.warn("Screen share cancelled or failed:", err);
  }
}

function stopLiveStream() {
  isLiveStreaming = false;
  liveStreamType = null;

  if (screenCamAnimId) {
    cancelAnimationFrame(screenCamAnimId);
    screenCamAnimId = null;
  }
  if (activeLiveStream) {
    activeLiveStream.getTracks().forEach(t => t.stop());
    activeLiveStream = null;
  }
  if (activeLiveVideo) {
    activeLiveVideo.pause();
    activeLiveVideo.srcObject = null;
    activeLiveVideo = null;
  }

  if (liveFeedImg) liveFeedImg.classList.remove("opacity-0");
  if (screenCamStatus) screenCamStatus.classList.add("hidden");
  if (screenCamMotionBadge) screenCamMotionBadge.classList.add("hidden");

  if (btnToggleWebcam) {
    btnToggleWebcam.className = "px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold rounded-xl transition flex items-center gap-1.5 shadow-lg shadow-emerald-600/20 active:scale-95";
    webcamIcon.textContent = "📹";
    webcamText.textContent = "Start Live Webcam";
  }
  if (btnToggleScreenCam) {
    btnToggleScreenCam.className = "px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-xl transition flex items-center gap-1.5 shadow-lg shadow-indigo-600/20 active:scale-95";
    screenCamIcon.textContent = "🖥️";
    screenCamText.textContent = "Start Screen Cam";
  }

  prevMotionData = null;
  renderRoiCanvas();
}

function screenCamLoop() {
  const targetCvs = roiCanvasGarden || roiCanvasCam1 || roiCanvas;
  if (!isLiveStreaming || !activeLiveVideo || !targetCvs) return;

  const ctx = targetCvs.getContext("2d");
  const w = targetCvs.width;
  const h = targetCvs.height;

  if (activeLiveVideo.videoWidth > 0 && activeLiveVideo.videoHeight > 0) {
    ctx.clearRect(0, 0, w, h);

    // 1. Calculate Digital Zoom & Pan Source Crop Coordinates
    const vWidth = activeLiveVideo.videoWidth;
    const vHeight = activeLiveVideo.videoHeight;
    const cropW = vWidth / currentZoom;
    const cropH = vHeight / currentZoom;
    const cropX = Math.max(0, Math.min(vWidth - cropW, (panX * vWidth) - (cropW / 2)));
    const cropY = Math.max(0, Math.min(vHeight - cropH, (panY * vHeight) - (cropH / 2)));

    // 2. Render Real-Time Video Frame to Canvas with dynamic vertical stretch
    if (isClarityBoostActive) {
      ctx.filter = "contrast(1.35) brightness(1.18) saturate(1.25)";
    } else {
      ctx.filter = "none";
    }
    ctx.save();
    ctx.translate(0, h * (1 - currentVerticalStretch) / 2);
    ctx.scale(1.0, currentVerticalStretch);
    ctx.drawImage(activeLiveVideo, cropX, cropY, cropW, cropH, 0, 0, w, h);
    ctx.restore();
    ctx.filter = "none";

    // 3. Motion Detection Engine (Runs every 150ms)
    const now = Date.now();
    if (now - lastMotionCheckTime > 150) {
      lastMotionCheckTime = now;
      checkScreenMotion(activeLiveVideo, cropX, cropY, cropW, cropH);
    }
    // 5. Render Detected Target Bounding Box & Perimeter (if active in last 6 seconds)
    if (lastDetectedTargetBox && (now - lastDetectedTargetTime < 6000)) {
      renderDetectedTargetBoundingBox(ctx, w, h, lastDetectedTargetBox, lastDetectedTargetLabel, lastDetectedTargetType, lastDetectedTargetConf);
    }

    // 4. Render Active Polygon Detection Zone Overlay on top of live video
    renderScreenZoneOverlay(ctx, w, h);
  }

  screenCamAnimId = requestAnimationFrame(screenCamLoop);
}

function renderScreenZoneOverlay(ctx, w, h) {
  if (!isZoneVisible && !isDrawingZone) return;

  if (activePolygon && activePolygon.length >= 3 && !isDrawingZone) {
    ctx.save();
    // Dim background outside zone
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.fillRect(0, 0, w, h);

    // Cut out active polygon
    ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath();
    ctx.moveTo(activePolygon[0].x * w, activePolygon[0].y * h);
    for (let i = 1; i < activePolygon.length; i++) {
      ctx.lineTo(activePolygon[i].x * w, activePolygon[i].y * h);
    }
    ctx.closePath();
    ctx.fill();

    ctx.globalCompositeOperation = "source-over";

    // Glowing border
    ctx.beginPath();
    ctx.moveTo(activePolygon[0].x * w, activePolygon[0].y * h);
    for (let i = 1; i < activePolygon.length; i++) {
      ctx.lineTo(activePolygon[i].x * w, activePolygon[i].y * h);
    }
    ctx.closePath();
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#f59e0b";
    ctx.shadowBlur = 8;
    ctx.stroke();

    // Zone tag
    const centerX = activePolygon.reduce((sum, p) => sum + p.x, 0) / activePolygon.length * w;
    const centerY = activePolygon.reduce((sum, p) => sum + p.y, 0) / activePolygon.length * h;
    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
    ctx.fillRect(centerX - 55, centerY - 12, 110, 22);
    ctx.strokeRect(centerX - 55, centerY - 12, 110, 22);
    ctx.fillStyle = "#fbbf24";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("🎯 Monitored Zone", centerX, centerY);
    ctx.restore();
  }
}

// --- Target Reticle Drawing System (Draws a crisp target positioned just above identified objects) ---
function drawTargetReticle(ctx, cx, cy, label, type, conf, strokeCol, tagEmoji, targetTopY) {
  ctx.save();

  // Target reticle radius & geometry
  const radius = 15;
  const innerRadius = 3;
  const color = strokeCol || "#ef4444";
  const emoji = tagEmoji || "🎯";
  const lbl = label || "Target";

  // Outer glowing circle
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;

  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Subtle translucent ring fill
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.18;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1.0;

  // Center bullseye dot
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, innerRadius, 0, Math.PI * 2);
  ctx.fill();

  // 4 Crosshair ticks (top, bottom, left, right)
  const tickLen = 7;
  const tickGap = 3;
  ctx.lineWidth = 2;
  ctx.beginPath();
  // Top
  ctx.moveTo(cx, cy - radius - tickGap);
  ctx.lineTo(cx, cy - radius - tickGap - tickLen);
  // Bottom
  ctx.moveTo(cx, cy + radius + tickGap);
  ctx.lineTo(cx, cy + radius + tickGap + tickLen);
  // Left
  ctx.moveTo(cx - radius - tickGap, cy);
  ctx.lineTo(cx - radius - tickGap - tickLen, cy);
  // Right
  ctx.moveTo(cx + radius + tickGap, cy);
  ctx.lineTo(cx + radius + tickGap + tickLen, cy);
  ctx.stroke();

  // Downward pointer tick connecting directly towards the rat top edge
  if (targetTopY && targetTopY > cy + radius) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(cx, cy + radius + tickGap + tickLen);
    ctx.lineTo(cx, Math.min(targetTopY, cy + radius + tickGap + tickLen + 14));
    ctx.stroke();
  }

  // Floating label badge positioned above the target reticle
  const confStr = conf ? ` ${Math.round(conf * 100)}%` : "";
  const tagText = `${emoji} ${lbl.toUpperCase()}${confStr}`;
  ctx.font = "bold 11px Inter, sans-serif";
  const textWidth = ctx.measureText(tagText).width;
  const badgeW = textWidth + 14;
  const badgeH = 20;
  const badgeX = cx - (badgeW / 2);
  const badgeY = Math.max(8, cy - radius - tickGap - tickLen - badgeH - 4);

  // Badge background rounded rectangle
  ctx.fillStyle = color;
  ctx.shadowColor = "rgba(0,0,0,0.6)";
  ctx.shadowBlur = 6;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 6);
  } else {
    ctx.rect(badgeX, badgeY, badgeW, badgeH);
  }
  ctx.fill();

  // Badge text
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(tagText, cx, badgeY + (badgeH / 2));

  ctx.restore();
}

function renderDetectedTargetBoundingBox(ctx, w, h, box, label, type, conf) {
  if (!box || box.length < 4) return;

  // ONLY bound an object when it could be an animal
  const nonAnimalTypes = ["clutter", "false_positive_clutter", "none", "shadow", "foliage", "clear", "manure", "horses_poo"];
  if (type && nonAnimalTypes.includes(type.toLowerCase())) {
    return;
  }

  // box: [ymin, xmin, ymax, xmax] (normalized 0-1000 or 0-1)
  const ymin = (box[0] > 1 ? box[0] / 1000 : box[0]) * h;
  const xmin = (box[1] > 1 ? box[1] / 1000 : box[1]) * w;
  const ymax = (box[2] > 1 ? box[2] / 1000 : box[2]) * h;
  const xmax = (box[3] > 1 ? box[3] / 1000 : box[3]) * w;

  const bw = Math.max(12, xmax - xmin);
  const bh = Math.max(12, ymax - ymin);

  const isTracking = (type === "tracking" || !conf);
  const meta = getObjectMeta(type, label);
  const strokeColor = isTracking ? "#38bdf8" : meta.color;

  ctx.save();

  // 1. Draw Object Boundary Rectangle (crisp 1.5px stroke)
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(xmin, ymin, bw, bh);

  // 2. Subtle translucent inner fill
  ctx.fillStyle = isTracking ? "rgba(56, 189, 248, 0.12)" : "rgba(239, 68, 68, 0.14)";
  ctx.fillRect(xmin, ymin, bw, bh);

  // 3. Four High-Contrast Corner Brackets
  const cornerLen = Math.min(14, Math.min(bw, bh) / 3);
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = strokeColor;

  // Top-left
  ctx.beginPath();
  ctx.moveTo(xmin, ymin + cornerLen);
  ctx.lineTo(xmin, ymin);
  ctx.lineTo(xmin + cornerLen, ymin);
  ctx.stroke();

  // Top-right
  ctx.beginPath();
  ctx.moveTo(xmax - cornerLen, ymin);
  ctx.lineTo(xmax, ymin);
  ctx.lineTo(xmax, ymin + cornerLen);
  ctx.stroke();

  // Bottom-left
  ctx.beginPath();
  ctx.moveTo(xmin, ymax - cornerLen);
  ctx.lineTo(xmin, ymax);
  ctx.lineTo(xmin + cornerLen, ymax);
  ctx.stroke();

  // Bottom-right
  ctx.beginPath();
  ctx.moveTo(xmax - cornerLen, ymax);
  ctx.lineTo(xmax, ymax);
  ctx.lineTo(xmax, ymax - cornerLen);
  ctx.stroke();

  // 4. Reticle & Floating AI Identification Tag
  const cx = (xmin + xmax) / 2;
  const reticleRadius = 14;
  let cy = ymin - reticleRadius - 10;
  if (cy < 32) {
    cy = ymin + reticleRadius + 6;
  }

  const displayLabel = isTracking ? "Candidate Object" : meta.label;
  const displayEmoji = isTracking ? "⚡" : meta.emoji;
  drawTargetReticle(ctx, cx, cy, displayLabel, type, conf, strokeColor, displayEmoji, ymin);

  ctx.restore();
}

function renderDetectedRatBoundingBox(ctx, w, h, box) {
  renderDetectedTargetBoundingBox(ctx, w, h, box, "Rat", "rat", 0.95);
}


function checkScreenMotion(video, cropX, cropY, cropW, cropH) {
  const motionW = 160;
  const motionH = 90;
  motionCanvas.width = motionW;
  motionCanvas.height = motionH;

  // Draw scaled down frame
  motionCtx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, motionW, motionH);
  const currentImgData = motionCtx.getImageData(0, 0, motionW, motionH);
  const data = currentImgData.data;

  if (!prevMotionData) {
    prevMotionData = data;
    return;
  }

  // Fast pixel difference calculation
  let changedPixels = 0;
  const threshold = 28; // Pixel intensity change threshold
  for (let i = 0; i < data.length; i += 4) {
    const diffR = Math.abs(data[i] - prevMotionData[i]);
    const diffG = Math.abs(data[i + 1] - prevMotionData[i + 1]);
    const diffB = Math.abs(data[i + 2] - prevMotionData[i + 2]);
    const maxDiff = Math.max(diffR, diffG, diffB);

    if (maxDiff > threshold) {
      changedPixels++;
    }
  }

  prevMotionData = data;

  const totalPixels = motionW * motionH;
  const motionPercent = (changedPixels / totalPixels) * 100;

  // Significant Motion Trigger Threshold: > 0.4% frame change
  if (motionPercent > 0.4) {
    if (screenCamMotionBadge) {
      screenCamMotionBadge.classList.remove("hidden");
      screenCamMotionBadge.textContent = `⚡ Motion ${motionPercent.toFixed(1)}%`;
    }

    const now = Date.now();
    // Throttle AI Vision checks: at most once every 2 seconds on motion
    if (!isGemmaAnalyzingScreen && (now - lastGemmaTriggerTime > 2000)) {
      lastGemmaTriggerTime = now;
      triggerScreenCamAiAnalysis();
    }
  } else {
    if (screenCamMotionBadge) {
      screenCamMotionBadge.classList.add("hidden");
    }
  }
}

async function triggerScreenCamAiAnalysis() {
  if (!activeLiveVideo || isGemmaAnalyzingScreen) return;
  isGemmaAnalyzingScreen = true;

  try {
    // Capture high-res snapshot from live stream with current zoom/pan crop applied
    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = activeLiveVideo.videoWidth || 1280;
    captureCanvas.height = activeLiveVideo.videoHeight || 720;
    const captureCtx = captureCanvas.getContext("2d");

    // Apply digital zoom crop
    const cropW = captureCanvas.width / currentZoom;
    const cropH = captureCanvas.height / currentZoom;
    const cropX = Math.max(0, Math.min(captureCanvas.width - cropW, panX * captureCanvas.width - cropW / 2));
    const cropY = Math.max(0, Math.min(captureCanvas.height - cropH, panY * captureCanvas.height - cropH / 2));

    captureCtx.drawImage(activeLiveVideo, cropX, cropY, cropW, cropH, 0, 0, captureCanvas.width, captureCanvas.height);
    const b64Data = captureCanvas.toDataURL("image/jpeg", 0.90);

    const polyArray = activePolygon && activePolygon.length >= 3
      ? activePolygon.map(p => [Number(p.x.toFixed(4)), Number(p.y.toFixed(4))])
      : null;

    const sourceName = liveStreamType === "webcam"
      ? `Local Webcam (${currentZoom.toFixed(1)}x Zoom)`
      : `Screen Cam (${currentZoom.toFixed(1)}x Zoom)`;

    const resp = await fetch("/api/screen_cam/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: b64Data,
        device_name: sourceName,
        polygon: polyArray
      })
    });

    const result = await resp.json();
    const isHit = result.detected || result.rat_detected;
    if (isHit) {
      const isPheasant = (result.object_type || "").toLowerCase() === "pheasant";
      const tagEmoji = isPheasant ? "🦚" : "🚨";
      const tagLabel = result.label || (isPheasant ? "Pheasant" : "Rat");

      lastDetectedTargetBox = result.bounding_box;
      lastDetectedTargetTime = Date.now();
      lastDetectedTargetLabel = tagLabel;
      lastDetectedTargetType = result.object_type || (isPheasant ? "pheasant" : "rat");
      lastDetectedTargetConf = result.confidence || 0.0;

      if (liveFrameAnalysisStatus) {
        liveFrameAnalysisStatus.textContent = `${tagEmoji} ${tagLabel.toUpperCase()} DETECTED! (${Math.round(result.confidence * 100)}%)`;
      }
      playAlertAudio();
      await fetchDetections();
    } else {
      if (liveFrameAnalysisStatus) {
        liveFrameAnalysisStatus.textContent = "✅ Scene clear (No target confirmed)";
      }
    }
  } catch (err) {
    console.error("Error during Screen Cam Gemma inference:", err);
  } finally {
    isGemmaAnalyzingScreen = false;
  }
}

// Initialize Screen Cam and Digital Zoom System
initScreenCamAndZoom();

// Initialize Vertical Un-Squash Presets
document.querySelectorAll(".btn-stretch-preset").forEach(btn => {
  btn.addEventListener("click", () => {
    const s = parseFloat(btn.dataset.stretch);
    if (!isNaN(s)) {
      setVerticalStretch(s);
    }
  });
});

