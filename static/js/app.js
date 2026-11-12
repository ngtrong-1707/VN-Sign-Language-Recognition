/**
 * VSL Vision Studio — Application Client Engine
 *
 * - Landmark extraction runs in the browser (MediaPipe Tasks Vision / WASM).
 * - Two transports, chosen by <body data-mode>:
 *     server     → per-frame landmarks over WebSocket /ws/stream (server keeps the 30-frame buffer)
 *     serverless → client-side 30-frame sliding window + POST /api/predict (Vercel)
 * - Fallback (server mode only): JPEG frames over WebSocket when MediaPipe WASM cannot load.
 * - 2D skeleton HUD, interactive sign simulation, video timeline, TTS, feedback.
 */

const MP_VERSION = '0.10.14';
const MP_BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`;
const MP_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const SEQ_LEN = 30;
const FEATURE_DIM = 258;
const POSE_DIM = 132;
const IDLE_RESET_FRAMES = 6;
const PREDICT_EVERY_N_FRAMES = 4;
const LIVE_FRAME_INTERVAL_MS = 40;
const VIDEO_SAMPLE_FPS = 20;
const VIDEO_MAX_SECONDS = 60;

const FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20]
];

class VSLApp {
    constructor() {
        this.mode = document.body.dataset.mode === 'serverless' ? 'serverless' : 'server';

        this.ws = null;
        this.video = document.getElementById('video-element');
        this.canvas = document.getElementById('canvas-element');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

        // Offscreen canvas: mirrored frame capture (training data was recorded on mirrored webcam frames)
        this.captureCanvas = document.createElement('canvas');
        this.captureCanvas.width = 640;
        this.captureCanvas.height = 360;
        this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: false });

        // MediaPipe (browser)
        this.handLandmarker = null;
        this.poseLandmarker = null;
        this.visionReady = false;
        this.visionPromise = null;
        this.lastMpTs = 0;

        // Live state
        this.isStreaming = false;
        this.isSimulating = false;
        this.simTimer = null;
        this.rafId = null;
        this.lastProcess = 0;
        this.processing = false;
        this.awaitingWs = false;
        this.isMirror = true;

        // Client-side buffer (serverless transport)
        this.clientBuffer = [];
        this.idleFrames = 0;
        this.framesSincePredict = 0;
        this.predictInFlight = false;

        // Telemetry
        this.fps = 0;
        this.frameCount = 0;
        this.lastFpsCalc = performance.now();

        // Sentence builder
        this.sentenceWords = [];
        this.lastConfidentSign = null;
        this.lastSignTimestamp = 0;
        this.currentRawPred = '';
        this.currentConfidence = 0;
        this.selectedDeviceId = null;

        this.initEventListeners();
        this.enumerateCameras();
        this.initTransport();
        this.initTTS();

        // Warm up MediaPipe in the background so the camera starts instantly
        setTimeout(() => this.initVision(), 1200);
    }

    // ------------------------------------------------------------------
    // UI wiring
    // ------------------------------------------------------------------
    initEventListeners() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchView(btn.dataset.view));
        });

        const btnToggleCam = document.getElementById('btn-toggle-cam');
        if (btnToggleCam) btnToggleCam.addEventListener('click', () => this.toggleCamera());

        const btnDemoSim = document.getElementById('btn-demo-sim');
        if (btnDemoSim) btnDemoSim.addEventListener('click', () => this.runDemoSequenceTour());

        document.querySelectorAll('.guide-tag[data-action]').forEach(tag => {
            tag.addEventListener('click', () => {
                if (tag.dataset.action) this.simulateGesture(tag.dataset.action);
            });
        });

        document.querySelectorAll('.btn-sim-card').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (btn.dataset.simAction) {
                    this.switchView('live');
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                    this.simulateGesture(btn.dataset.simAction);
                }
            });
        });

        const btnDemoTimeline = document.getElementById('btn-demo-timeline');
        if (btnDemoTimeline) btnDemoTimeline.addEventListener('click', () => this.runDemoTimeline());

        const camSelect = document.getElementById('select-camera-source');
        if (camSelect) {
            camSelect.addEventListener('change', async (e) => {
                this.selectedDeviceId = e.target.value;
                if (this.isStreaming) {
                    await this.toggleCamera();
                    await this.toggleCamera();
                }
            });
        }

        const btnMirror = document.getElementById('btn-mirror');
        if (btnMirror) {
            btnMirror.addEventListener('click', () => {
                this.isMirror = !this.isMirror;
                if (this.video) this.video.style.transform = this.isMirror ? 'scaleX(-1)' : 'scaleX(1)';
            });
        }

        const btnSpeak = document.getElementById('btn-speak-sentence');
        if (btnSpeak) btnSpeak.addEventListener('click', () => this.speakSentence());

        const btnClear = document.getElementById('btn-clear-sentence');
        if (btnClear) {
            btnClear.addEventListener('click', () => {
                this.sentenceWords = [];
                this.lastConfidentSign = null;
                this.renderSentence();
            });
        }

        const btnCopy = document.getElementById('btn-copy-sentence');
        if (btnCopy) {
            btnCopy.addEventListener('click', () => {
                const text = this.sentenceWords.join(' ');
                if (text && navigator.clipboard) {
                    navigator.clipboard.writeText(text);
                    btnCopy.textContent = 'Đã chép!';
                    setTimeout(() => btnCopy.textContent = 'Sao chép', 2000);
                }
            });
        }

        const btnFeedback = document.getElementById('btn-feedback-trigger');
        const modal = document.getElementById('feedback-modal');
        const btnCloseModal = document.getElementById('btn-close-modal');
        const btnSubmitFeedback = document.getElementById('btn-submit-feedback');

        if (btnFeedback && modal) {
            btnFeedback.addEventListener('click', () => {
                const predInput = document.getElementById('feedback-pred');
                if (predInput) predInput.value = this.currentRawPred || 'Chưa có dự đoán';
                modal.classList.remove('hidden');
            });
        }
        if (btnCloseModal && modal) btnCloseModal.addEventListener('click', () => modal.classList.add('hidden'));
        if (modal) modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });
        if (btnSubmitFeedback) btnSubmitFeedback.addEventListener('click', () => this.submitFeedback());

        const dictSearch = document.getElementById('dict-search');
        if (dictSearch) dictSearch.addEventListener('input', (e) => this.filterDictionary(e.target.value));

        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                this.filterDictionaryByCategory(chip.dataset.cat);
            });
        });

        const dropzone = document.getElementById('upload-dropzone');
        const fileInput = document.getElementById('video-file-input');
        if (dropzone && fileInput) {
            dropzone.addEventListener('click', () => fileInput.click());
            dropzone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
            dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
            dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) this.processUploadedVideo(e.dataTransfer.files[0]);
            });
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) this.processUploadedVideo(e.target.files[0]);
                fileInput.value = '';
            });
        }
    }

    switchView(viewId) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.view-section').forEach(v => v.classList.remove('active'));

        const targetBtn = document.querySelector(`.tab-btn[data-view="${viewId}"]`);
        const targetView = document.getElementById(`view-${viewId}`);
        if (targetBtn && targetView) {
            targetBtn.classList.add('active');
            targetView.classList.add('active');
        }
    }

    setStatus(text, online) {
        const statusText = document.getElementById('ws-status-text');
        const statusDot = document.querySelector('.status-dot');
        if (statusText) statusText.textContent = text;
        if (statusDot) statusDot.className = online ? 'status-dot online' : 'status-dot';
    }

    showToast(text) {
        const toast = document.getElementById('engine-toast');
        if (!toast) return;
        if (text) {
            toast.textContent = text;
            toast.classList.remove('hidden');
        } else {
            toast.classList.add('hidden');
        }
    }

    // ------------------------------------------------------------------
    // Transport (WebSocket for local server, HTTP for serverless)
    // ------------------------------------------------------------------
    initTransport() {
        if (this.mode === 'server') {
            this.connectWebSocket();
            return;
        }
        fetch('/api/health')
            .then(r => r.ok ? r.json() : Promise.reject(new Error(r.statusText)))
            .then(() => this.setStatus('ONNX · SERVERLESS', true))
            .catch(() => this.setStatus('OFFLINE', false));
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);

        this.ws.onopen = () => {
            this.awaitingWs = false;
            this.setStatus('ONNX ENGINE', true);
        };
        this.ws.onmessage = (event) => {
            this.awaitingWs = false;
            this.handlePredictionMessage(JSON.parse(event.data));
        };
        this.ws.onclose = () => {
            this.awaitingWs = false;
            this.setStatus('OFFLINE', false);
            setTimeout(() => this.connectWebSocket(), 2000);
        };
    }

    wsReady() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    resetBuffers() {
        this.clientBuffer = [];
        this.idleFrames = 0;
        this.framesSincePredict = 0;
        if (this.wsReady()) this.ws.send(JSON.stringify({ type: 'reset' }));
    }

    // ------------------------------------------------------------------
    // MediaPipe Tasks Vision (browser-side landmark extraction)
    // ------------------------------------------------------------------
    initVision() {
        if (this.visionReady) return Promise.resolve(true);
        if (this.visionPromise) return this.visionPromise;

        this.visionPromise = (async () => {
            const vision = await import(MP_BUNDLE_URL);
            const fileset = await vision.FilesetResolver.forVisionTasks(MP_WASM_URL);

            const create = async (TaskClass, options) => {
                try {
                    return await TaskClass.createFromOptions(fileset, {
                        ...options,
                        baseOptions: { ...options.baseOptions, delegate: 'GPU' }
                    });
                } catch (gpuErr) {
                    console.warn('[MEDIAPIPE] GPU delegate unavailable, falling back to CPU:', gpuErr);
                    return await TaskClass.createFromOptions(fileset, {
                        ...options,
                        baseOptions: { ...options.baseOptions, delegate: 'CPU' }
                    });
                }
            };

            this.handLandmarker = await create(vision.HandLandmarker, {
                baseOptions: { modelAssetPath: HAND_MODEL_URL },
                runningMode: 'VIDEO',
                numHands: 2,
                minHandDetectionConfidence: 0.5,
                minHandPresenceConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            this.poseLandmarker = await create(vision.PoseLandmarker, {
                baseOptions: { modelAssetPath: POSE_MODEL_URL },
                runningMode: 'VIDEO',
                numPoses: 1,
                minPoseDetectionConfidence: 0.5,
                minPosePresenceConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            this.visionReady = true;
            console.info('[MEDIAPIPE] Tasks Vision ready (browser-side landmark extraction).');
            return true;
        })().catch(err => {
            console.warn('[MEDIAPIPE] Không thể tải MediaPipe WASM:', err);
            this.visionPromise = null;
            return false;
        });

        return this.visionPromise;
    }

    nextTimestamp() {
        const now = performance.now();
        this.lastMpTs = now > this.lastMpTs ? now : this.lastMpTs + 1;
        return this.lastMpTs;
    }

    /** Draws `source` mirrored onto the capture canvas (matches training-time cv2.flip). */
    captureMirrored(source, srcW, srcH) {
        const cw = this.captureCanvas.width;
        const ch = Math.round(cw * (srcH / srcW)) || 360;
        if (this.captureCanvas.height !== ch) this.captureCanvas.height = ch;

        this.captureCtx.save();
        this.captureCtx.translate(cw, 0);
        this.captureCtx.scale(-1, 1);
        this.captureCtx.drawImage(source, 0, 0, cw, ch);
        this.captureCtx.restore();
    }

    /** Runs Pose + Hand landmarkers on the capture canvas and returns a 258-d vector. */
    extractFeatureVector() {
        const ts = this.nextTimestamp();
        const poseRes = this.poseLandmarker.detectForVideo(this.captureCanvas, ts);
        const handRes = this.handLandmarker.detectForVideo(this.captureCanvas, ts);

        const vec = new Array(FEATURE_DIM).fill(0);
        const r = (v) => Math.round(v * 1e5) / 1e5;

        if (poseRes.landmarks && poseRes.landmarks.length > 0) {
            poseRes.landmarks[0].slice(0, 33).forEach((lm, i) => {
                vec[i * 4] = r(lm.x);
                vec[i * 4 + 1] = r(lm.y);
                vec[i * 4 + 2] = r(lm.z);
                vec[i * 4 + 3] = r(lm.visibility ?? 0.99);
            });
        }

        const handedness = handRes.handedness || handRes.handednesses || [];
        (handRes.landmarks || []).forEach((lms, idx) => {
            const label = (handedness[idx] && handedness[idx][0] && handedness[idx][0].categoryName) || 'Right';
            const offset = label === 'Left' ? POSE_DIM : POSE_DIM + 63;
            lms.slice(0, 21).forEach((lm, j) => {
                vec[offset + j * 3] = r(lm.x);
                vec[offset + j * 3 + 1] = r(lm.y);
                vec[offset + j * 3 + 2] = r(lm.z);
            });
        });

        return vec;
    }

    static hasHands(vec) {
        for (let i = POSE_DIM; i < FEATURE_DIM; i++) {
            if (vec[i] !== 0) return true;
        }
        return false;
    }

    /** JS port of `extract_2d_overlay` (vsl/serverless/core.py). */
    static buildOverlay(vec) {
        const hands = [];
        [['Left', POSE_DIM], ['Right', POSE_DIM + 63]].forEach(([label, start]) => {
            const pts = [];
            let any = false;
            for (let j = 0; j < 21; j++) {
                const x = vec[start + j * 3];
                const y = vec[start + j * 3 + 1];
                if (x !== 0 || y !== 0) any = true;
                pts.push([x, y]);
            }
            if (any) hands.push({ label, points: pts });
        });

        const pose = [];
        [11, 12, 13, 14, 15, 16].forEach(idx => {
            const x = vec[idx * 4];
            const y = vec[idx * 4 + 1];
            if (x > 0 || y > 0) pose.push([x, y]);
        });

        return { hands, pose };
    }

    // ------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------
    async enumerateCameras() {
        const camSelect = document.getElementById('select-camera-source');
        if (!camSelect) return;

        try {
            if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(d => d.kind === 'videoinput');

                camSelect.innerHTML = '';
                if (videoDevices.length === 0) {
                    camSelect.innerHTML = '<option value="">Camera Mặc Định</option>';
                    return;
                }

                videoDevices.forEach((device, idx) => {
                    const opt = document.createElement('option');
                    opt.value = device.deviceId;
                    opt.textContent = device.label || `Camera ${idx + 1}`;
                    if (device.deviceId === this.selectedDeviceId) opt.selected = true;
                    camSelect.appendChild(opt);
                });

                if (!this.selectedDeviceId) this.selectedDeviceId = videoDevices[0].deviceId;
            } else {
                camSelect.innerHTML = '<option value="">Camera Mặc Định</option>';
            }
        } catch (err) {
            console.warn('[CAMERA] Không thể truy vấn danh sách thiết bị:', err);
            camSelect.innerHTML = '<option value="">Camera Mặc Định</option>';
        }
    }

    stopSimulation() {
        if (this.simTimer) {
            clearInterval(this.simTimer);
            this.simTimer = null;
        }
        this.isSimulating = false;
    }

    async toggleCamera() {
        this.stopSimulation();
        const placeholder = document.getElementById('viewport-placeholder');
        const btnToggleCam = document.getElementById('btn-toggle-cam');
        const hudBadge = document.getElementById('hud-badge');

        if (!this.isStreaming) {
            try {
                const constraints = {
                    video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
                    audio: false
                };
                if (this.selectedDeviceId) constraints.video.deviceId = { exact: this.selectedDeviceId };

                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                this.video.srcObject = stream;
                this.video.muted = true;
                await this.video.play();

                this.canvas.width = this.video.videoWidth || 1280;
                this.canvas.height = this.video.videoHeight || 720;

                this.isStreaming = true;
                this.awaitingWs = false;
                if (placeholder) placeholder.classList.add('hidden');
                if (hudBadge) hudBadge.classList.remove('hidden');
                if (btnToggleCam) {
                    btnToggleCam.textContent = '⏹ Dừng Camera';
                    btnToggleCam.classList.replace('btn-primary', 'btn-secondary');
                }

                this.resetBuffers();
                await this.enumerateCameras();

                if (!this.visionReady) {
                    this.showToast('Đang tải MediaPipe (WASM)...');
                    this.initVision().then(ok => {
                        if (ok) this.showToast('');
                        else if (this.mode === 'server') this.showToast('MediaPipe WASM lỗi → dùng nhận diện phía máy chủ');
                        else this.showToast('Không tải được MediaPipe. Kiểm tra kết nối mạng.');
                    });
                }

                this.lastProcess = 0;
                this.rafId = requestAnimationFrame(() => this.liveLoop());
            } catch (err) {
                alert(`Không thể truy cập camera: ${err.message}\nBạn có thể nhấn "▶ Chạy Mô Phỏng AI (Demo)" hoặc chọn một cử chỉ bên phải để kiểm thử!`);
            }
        } else {
            if (this.video && this.video.srcObject) {
                this.video.srcObject.getTracks().forEach(track => track.stop());
                this.video.srcObject = null;
            }
            if (this.rafId) cancelAnimationFrame(this.rafId);
            this.rafId = null;
            this.isStreaming = false;
            this.showToast('');
            if (this.ctx) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            if (placeholder) placeholder.classList.remove('hidden');
            if (hudBadge) hudBadge.classList.add('hidden');
            if (btnToggleCam) {
                btnToggleCam.textContent = '📷 Bật Camera';
                btnToggleCam.classList.replace('btn-secondary', 'btn-primary');
            }
        }
    }

    liveLoop() {
        if (!this.isStreaming) return;
        const now = performance.now();
        if (!this.processing && now - this.lastProcess >= LIVE_FRAME_INTERVAL_MS && this.video.readyState >= 2) {
            this.lastProcess = now;
            this.processing = true;
            try {
                this.processLiveFrame();
            } catch (err) {
                console.error('[LIVE]', err);
            } finally {
                this.processing = false;
            }
        }
        this.rafId = requestAnimationFrame(() => this.liveLoop());
    }

    processLiveFrame() {
        const vw = this.video.videoWidth || 1280;
        const vh = this.video.videoHeight || 720;
        this.captureMirrored(this.video, vw, vh);

        if (this.visionReady) {
            const vec = this.extractFeatureVector();
            this.drawLandmarks2D(VSLApp.buildOverlay(vec));

            if (this.mode === 'server' && this.wsReady()) {
                this.ws.send(JSON.stringify({ landmarks: vec }));
            } else if (this.mode === 'serverless') {
                this.pushClientFrame(vec);
            }
            this.tickFps();
        } else if (this.mode === 'server' && this.wsReady() && !this.awaitingWs && this.visionPromise === null) {
            // Fallback: server-side MediaPipe on JPEG frames
            this.awaitingWs = true;
            this.ws.send(JSON.stringify({ image: this.captureCanvas.toDataURL('image/jpeg', 0.7) }));
            this.tickFps();
        }
    }

    tickFps() {
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsCalc >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsCalc));
            this.frameCount = 0;
            this.lastFpsCalc = now;
            const fpsEl = document.getElementById('tele-fps');
            if (fpsEl) fpsEl.textContent = `${this.fps} FPS`;
        }
    }

    /** Serverless transport: client-side sliding window + idle detection + throttled POST /api/predict. */
    pushClientFrame(vec) {
        if (!VSLApp.hasHands(vec)) {
            this.idleFrames++;
            if (this.idleFrames >= IDLE_RESET_FRAMES) this.clientBuffer = [];
            this.handlePredictionMessage({ type: 'idle', buffer_size: this.clientBuffer.length, required: SEQ_LEN });
            return;
        }

        this.idleFrames = 0;
        this.clientBuffer.push(vec);
        if (this.clientBuffer.length > SEQ_LEN) this.clientBuffer.shift();

        if (this.clientBuffer.length < SEQ_LEN) {
            this.handlePredictionMessage({ type: 'buffering', buffer_size: this.clientBuffer.length, required: SEQ_LEN });
            return;
        }

        this.framesSincePredict++;
        if (this.framesSincePredict < PREDICT_EVERY_N_FRAMES || this.predictInFlight) return;

        this.framesSincePredict = 0;
        this.predictInFlight = true;
        const t0 = performance.now();
        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frames: this.clientBuffer })
        })
            .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
            .then(data => {
                if (!this.isStreaming) return;
                data.roundtrip_ms = Math.round(performance.now() - t0);
                this.handlePredictionMessage(data);
            })
            .catch(err => console.warn('[PREDICT]', err))
            .finally(() => { this.predictInFlight = false; });
    }

    // ------------------------------------------------------------------
    // HUD rendering
    // ------------------------------------------------------------------
    drawLandmarks2D(landmarks2d, isSimulation = false, simLabel = '') {
        if (!this.ctx || !this.canvas) return;
        const w = this.canvas.width || 640;
        const h = this.canvas.height || 360;
        // Landmarks live in mirrored-image space; the overlay is never CSS-flipped.
        const mapX = (x) => (this.isMirror ? x : 1 - x) * w;

        this.ctx.clearRect(0, 0, w, h);

        if (isSimulation) {
            this.ctx.fillStyle = '#090d16';
            this.ctx.fillRect(0, 0, w, h);
            this.ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
            this.ctx.lineWidth = 1;
            for (let x = 0; x < w; x += 32) {
                this.ctx.beginPath(); this.ctx.moveTo(x, 0); this.ctx.lineTo(x, h); this.ctx.stroke();
            }
            for (let y = 0; y < h; y += 32) {
                this.ctx.beginPath(); this.ctx.moveTo(0, y); this.ctx.lineTo(w, y); this.ctx.stroke();
            }
            if (simLabel) {
                this.ctx.fillStyle = 'rgba(56, 189, 248, 0.85)';
                this.ctx.font = '600 13px "JetBrains Mono", monospace';
                this.ctx.fillText(`AI SIMULATION MODE // ${simLabel.toUpperCase()}`, 18, h - 18);
            }
        }

        if (!landmarks2d) return;

        if (Array.isArray(landmarks2d.pose) && landmarks2d.pose.length > 0) {
            const posePts = landmarks2d.pose.map(p => [mapX(p[0]), p[1] * h]);
            this.ctx.strokeStyle = 'rgba(236, 72, 153, 0.55)';
            this.ctx.lineWidth = 2.5;
            if (posePts.length >= 2) {
                this.ctx.beginPath();
                this.ctx.moveTo(posePts[0][0], posePts[0][1]);
                this.ctx.lineTo(posePts[1][0], posePts[1][1]);
                this.ctx.stroke();
            }
            posePts.forEach(([px, py]) => {
                this.ctx.beginPath();
                this.ctx.arc(px, py, 5, 0, Math.PI * 2);
                this.ctx.fillStyle = '#ec4899';
                this.ctx.fill();
            });
        }

        if (Array.isArray(landmarks2d.hands)) {
            landmarks2d.hands.forEach(hand => {
                const pts = (hand.points || []).map(p => [mapX(p[0]), p[1] * h]);
                if (pts.length < 21) return;

                this.ctx.strokeStyle = hand.label === 'Left' ? '#38bdf8' : '#10b981';
                this.ctx.lineWidth = 2.5;
                FINGER_CHAINS.forEach(chain => {
                    for (let i = 0; i < chain.length - 1; i++) {
                        const p1 = pts[chain[i]];
                        const p2 = pts[chain[i + 1]];
                        this.ctx.beginPath();
                        this.ctx.moveTo(p1[0], p1[1]);
                        this.ctx.lineTo(p2[0], p2[1]);
                        this.ctx.stroke();
                    }
                });

                pts.forEach(([px, py], idx) => {
                    this.ctx.beginPath();
                    this.ctx.arc(px, py, idx === 0 ? 5.5 : 4, 0, Math.PI * 2);
                    this.ctx.fillStyle = '#00f0ff';
                    this.ctx.fill();
                    this.ctx.lineWidth = 1.5;
                    this.ctx.strokeStyle = '#a855f7';
                    this.ctx.stroke();
                });
            });
        }
    }

    // ------------------------------------------------------------------
    // Simulation (replays recorded dataset sequences)
    // ------------------------------------------------------------------
    async simulateGesture(actionName) {
        if (this.isStreaming) await this.toggleCamera();
        this.stopSimulation();

        const placeholder = document.getElementById('viewport-placeholder');
        const hudBadge = document.getElementById('hud-badge');

        try {
            const resp = await fetch(`/api/predict_video/demo_sequence/${encodeURIComponent(actionName)}`);
            if (!resp.ok) throw new Error('Không tìm thấy chuỗi mẫu');
            const data = await resp.json();

            this.canvas.width = 1280;
            this.canvas.height = 720;
            if (placeholder) placeholder.classList.add('hidden');
            if (hudBadge) hudBadge.classList.remove('hidden');

            this.isSimulating = true;
            const frames = data.frames || [];
            const overlays = data.overlays || [];
            let idx = 0;

            this.resetBuffers();
            const fpsEl = document.getElementById('tele-fps');
            if (fpsEl) fpsEl.textContent = '30 FPS';

            return new Promise((resolve) => {
                this.simTimer = setInterval(() => {
                    if (idx >= frames.length) {
                        clearInterval(this.simTimer);
                        this.simTimer = null;
                        if (data.prediction) {
                            this.handlePredictionMessage({ ...data.prediction, type: 'prediction' });
                        }
                        resolve();
                        return;
                    }

                    this.drawLandmarks2D(overlays[idx], true, actionName);
                    const bufEl = document.getElementById('tele-buffer');
                    if (bufEl) bufEl.textContent = `${idx + 1} / ${SEQ_LEN}`;

                    // Locally, also stream through the real WebSocket pipeline
                    if (this.mode === 'server' && this.wsReady()) {
                        this.ws.send(JSON.stringify({ landmarks: frames[idx] }));
                    }
                    idx++;
                }, 35);
            });
        } catch (err) {
            console.error('[SIMULATION ERROR]', err);
            this.showToast(`Lỗi mô phỏng: ${err.message}`);
            setTimeout(() => this.showToast(''), 2500);
        }
    }

    async runDemoSequenceTour() {
        const demoSigns = ['Xin chao', 'Toi', 'Yeu', 'Viet Nam', 'Cam on'];
        for (const sign of demoSigns) {
            await this.simulateGesture(sign);
            if (!this.isSimulating) break;
            await new Promise(r => setTimeout(r, 550));
            if (!this.isSimulating) break;
        }
    }

    // ------------------------------------------------------------------
    // Prediction handling
    // ------------------------------------------------------------------
    handlePredictionMessage(data) {
        // Server-side fallback mode returns its own landmarks for the overlay
        if (data.landmarks_2d && !this.isSimulating && !this.visionReady) {
            this.drawLandmarks2D(data.landmarks_2d, false);
        }

        const bufEl = document.getElementById('tele-buffer');
        const predNameEl = document.getElementById('hud-pred-name');
        const predConfEl = document.getElementById('hud-pred-conf');
        const predIconEl = document.getElementById('hud-pred-icon');
        const hudBadge = document.getElementById('hud-badge');

        if (data.type === 'prediction') {
            this.currentRawPred = data.name;
            this.currentConfidence = data.confidence;

            if (predNameEl) predNameEl.textContent = data.display;
            if (predConfEl) predConfEl.textContent = `${(data.confidence * 100).toFixed(0)}%`;
            if (predIconEl) predIconEl.textContent = data.icon;
            if (hudBadge) hudBadge.classList.toggle('confident', !!data.is_confident);

            const latEl = document.getElementById('tele-latency');
            if (latEl) latEl.textContent = `${data.latency_ms} ms`;
            if (bufEl) bufEl.textContent = `${data.buffer_size} / ${SEQ_LEN}`;

            this.renderProbabilityBars(data.top5);

            if (data.is_confident) {
                const now = performance.now();
                if (data.name !== this.lastConfidentSign || (now - this.lastSignTimestamp > 1800)) {
                    this.lastConfidentSign = data.name;
                    this.lastSignTimestamp = now;
                    this.sentenceWords.push(data.display);
                    if (this.sentenceWords.length > 8) this.sentenceWords.shift();
                    this.renderSentence();
                }
            }
        } else if (data.type === 'buffering') {
            if (bufEl) bufEl.textContent = `${data.buffer_size} / ${data.required}`;
        } else if (data.type === 'idle') {
            if (bufEl) bufEl.textContent = `${data.buffer_size} / ${data.required}`;
            if (predNameEl) predNameEl.textContent = 'Đang chờ cử chỉ tay...';
            if (predConfEl) predConfEl.textContent = '--';
            if (predIconEl) predIconEl.textContent = '✋';
            if (hudBadge) hudBadge.classList.remove('confident');
            // Allow the same sign to be appended again after a pause
            this.lastConfidentSign = null;
        }
    }

    renderProbabilityBars(top5) {
        const container = document.getElementById('prob-bars-container');
        if (!container || !top5) return;

        container.innerHTML = top5.map((item, idx) => {
            const pct = (item.probability * 100).toFixed(1);
            const active = idx === 0 ? 'active' : '';
            return `
                <div class="prob-item">
                    <div class="prob-item-header">
                        <span class="prob-item-name ${active}">${item.icon} ${item.display}</span>
                        <span class="prob-item-val ${active}">${pct}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-bar ${active}" style="width: ${pct}%"></div>
                    </div>
                </div>`;
        }).join('');
    }

    renderSentence() {
        const box = document.getElementById('sentence-box');
        if (!box) return;

        if (this.sentenceWords.length === 0) {
            box.innerHTML = '<span class="placeholder-text">Chưa ghi nhận cử chỉ nào. Hãy thực hiện ký hiệu trong khung hình...</span>';
        } else {
            box.innerHTML = this.sentenceWords.map(word => `<span class="sentence-word">${word}</span>`).join('');
        }
    }

    // ------------------------------------------------------------------
    // TTS & Feedback
    // ------------------------------------------------------------------
    initTTS() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = () => {
                this.voices = window.speechSynthesis.getVoices();
            };
        }
    }

    speakSentence() {
        const text = this.sentenceWords.join(' ');
        if (!text || !('speechSynthesis' in window)) return;

        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'vi-VN';
        utterance.rate = 0.95;
        const viVoice = window.speechSynthesis.getVoices().find(v => v.lang.includes('vi') || v.name.includes('Vietnamese'));
        if (viVoice) utterance.voice = viVoice;
        window.speechSynthesis.speak(utterance);
    }

    async submitFeedback() {
        const pred = document.getElementById('feedback-pred').value;
        const actual = document.getElementById('feedback-actual').value;
        const notes = document.getElementById('feedback-notes').value;
        const modal = document.getElementById('feedback-modal');

        try {
            const resp = await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    predicted_action: pred,
                    actual_action: actual,
                    confidence: this.currentConfidence || 0,
                    notes: notes
                })
            });
            const res = await resp.json();
            if (res.status === 'success') {
                alert('Cảm ơn bạn! Phản hồi đã được ghi nhận.');
                if (modal) modal.classList.add('hidden');
            }
        } catch (e) {
            alert('Lỗi gửi phản hồi: ' + e.message);
        }
    }

    // ------------------------------------------------------------------
    // Dictionary
    // ------------------------------------------------------------------
    filterDictionary(query) {
        const term = query.toLowerCase().trim();
        document.querySelectorAll('.dict-card').forEach(card => {
            const hay = `${card.dataset.name} ${card.dataset.desc} ${card.dataset.cat}`.toLowerCase();
            card.style.display = hay.includes(term) ? 'flex' : 'none';
        });
    }

    filterDictionaryByCategory(category) {
        document.querySelectorAll('.dict-card').forEach(card => {
            const show = !category || category === 'all' || card.dataset.cat === category;
            card.style.display = show ? 'flex' : 'none';
        });
    }

    // ------------------------------------------------------------------
    // Video analysis
    // ------------------------------------------------------------------
    setUploadStatus(text) {
        const progressEl = document.getElementById('upload-status');
        if (!progressEl) return;
        progressEl.classList.remove('hidden');
        progressEl.textContent = text;
    }

    renderTimelineResult(data, label) {
        const resultsEl = document.getElementById('upload-results');
        const sentenceEl = document.getElementById('upload-sentence');
        const timelineBody = document.getElementById('upload-timeline-body');

        this.setUploadStatus(`Phân tích hoàn tất! (${data.total_frames} frames / ${data.duration_sec}s${label ? ' — ' + label : ''})`);
        if (resultsEl) resultsEl.classList.remove('hidden');
        if (sentenceEl) sentenceEl.textContent = data.sentence;
        if (timelineBody) {
            timelineBody.innerHTML = (data.timeline || []).map(item => `
                <tr>
                    <td><code>${item.timestamp}</code></td>
                    <td><strong>${item.icon} ${item.display}</strong></td>
                    <td><span style="color: var(--accent-emerald); font-family: var(--font-mono);">${item.confidence}%</span></td>
                </tr>`).join('');
        }
    }

    async runDemoTimeline() {
        this.setUploadStatus('Đang chạy suy luận ONNX trên chuỗi 5 cử chỉ mẫu (150 frames)...');
        try {
            const resp = await fetch('/api/predict_video/demo_timeline', { method: 'POST' });
            const data = await resp.json();
            if (data.status === 'success') this.renderTimelineResult(data, 'ONNX Runtime');
        } catch (err) {
            this.setUploadStatus(`Lỗi chạy demo: ${err.message}`);
        }
    }

    async processUploadedVideo(file) {
        if (this.mode === 'serverless') return this.analyzeVideoInBrowser(file);

        this.setUploadStatus('Đang phân tích video qua mô hình VSL...');
        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/predict_video', { method: 'POST', body: formData });
            const data = await resp.json();
            if (data.status === 'success') this.renderTimelineResult(data);
            else this.setUploadStatus('Không thể nhận diện video.');
        } catch (err) {
            this.setUploadStatus(`Lỗi tải lên: ${err.message}`);
        }
    }

    /** Serverless: extract landmarks in the browser, then POST /api/predict_timeline. */
    async analyzeVideoInBrowser(file) {
        if (this.isStreaming) await this.toggleCamera();
        this.stopSimulation();

        this.setUploadStatus('Đang tải MediaPipe (WASM)...');
        const ok = await this.initVision();
        if (!ok) {
            this.setUploadStatus('Không tải được MediaPipe WASM. Vui lòng kiểm tra kết nối mạng.');
            return;
        }

        const url = URL.createObjectURL(file);
        const v = document.createElement('video');
        v.muted = true;
        v.playsInline = true;
        v.preload = 'auto';
        v.src = url;

        try {
            await new Promise((resolve, reject) => {
                v.onloadeddata = resolve;
                v.onerror = () => reject(new Error('Trình duyệt không đọc được định dạng video này (hãy dùng MP4/WebM).'));
            });

            const duration = Math.min(v.duration || 0, VIDEO_MAX_SECONDS);
            if (!duration) throw new Error('Video rỗng hoặc không xác định được thời lượng.');

            const total = Math.floor(duration * VIDEO_SAMPLE_FPS);
            const frames = [];
            const seek = (t) => new Promise(res => {
                const done = () => { v.removeEventListener('seeked', done); res(); };
                v.addEventListener('seeked', done);
                v.currentTime = t;
            });

            for (let i = 0; i < total; i++) {
                await seek(i / VIDEO_SAMPLE_FPS);
                this.captureMirrored(v, v.videoWidth || 640, v.videoHeight || 360);
                frames.push(this.extractFeatureVector());
                if (i % 10 === 0) {
                    this.setUploadStatus(`Đang trích xuất landmarks trên trình duyệt: ${Math.round((i / total) * 100)}% (${i}/${total} frames)`);
                }
            }

            this.setUploadStatus('Đang suy luận chuỗi ký hiệu (ONNX Runtime)...');
            const resp = await fetch('/api/predict_timeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames, fps: VIDEO_SAMPLE_FPS })
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            this.renderTimelineResult(data, `MediaPipe WASM @ ${VIDEO_SAMPLE_FPS} FPS`);
        } catch (err) {
            this.setUploadStatus(`Lỗi phân tích video: ${err.message}`);
        } finally {
            URL.revokeObjectURL(url);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.vslApp = new VSLApp();
});
