/**
 * VSL Vision - Real-time Vietnamese Sign Language Application Engine
 * WebSocket Streaming, HTML5 Video Ingestion, Web Speech TTS & UI State Management
 */

class VSLApp {
    constructor() {
        this.ws = null;
        this.video = document.getElementById('video-element');
        this.canvas = document.getElementById('canvas-element');
        this.ctx = this.canvas.getContext('2d');
        
        this.isStreaming = false;
        this.isMirror = true;
        this.streamInterval = null;
        this.fps = 0;
        this.frameCount = 0;
        this.lastFpsCalc = performance.now();

        this.sentenceWords = [];
        this.lastConfidentSign = null;
        this.lastSignTimestamp = 0;

        this.initEventListeners();
        this.connectWebSocket();
        this.initTTS();
    }

    initEventListeners() {
        // Tab Navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const targetView = btn.dataset.view;
                this.switchView(targetView);
            });
        });

        // Camera Controls
        const btnToggleCam = document.getElementById('btn-toggle-cam');
        if (btnToggleCam) {
            btnToggleCam.addEventListener('click', () => this.toggleCamera());
        }

        const btnMirror = document.getElementById('btn-mirror');
        if (btnMirror) {
            btnMirror.addEventListener('click', () => {
                this.isMirror = !this.isMirror;
                this.video.style.transform = this.isMirror ? 'scaleX(-1)' : 'scaleX(1)';
                this.canvas.style.transform = this.isMirror ? 'scaleX(-1)' : 'scaleX(1)';
            });
        }

        // Sentence Actions
        const btnSpeak = document.getElementById('btn-speak-sentence');
        if (btnSpeak) {
            btnSpeak.addEventListener('click', () => this.speakSentence());
        }

        const btnClear = document.getElementById('btn-clear-sentence');
        if (btnClear) {
            btnClear.addEventListener('click', () => {
                this.sentenceWords = [];
                this.renderSentence();
            });
        }

        const btnCopy = document.getElementById('btn-copy-sentence');
        if (btnCopy) {
            btnCopy.addEventListener('click', () => {
                const text = this.sentenceWords.join(' ');
                if (text) {
                    navigator.clipboard.writeText(text);
                    btnCopy.innerHTML = '✅ Đã sao chép!';
                    setTimeout(() => btnCopy.innerHTML = '📋 Sao chép', 2000);
                }
            });
        }

        // Dictionary Search
        const dictSearch = document.getElementById('dict-search');
        if (dictSearch) {
            dictSearch.addEventListener('input', (e) => this.filterDictionary(e.target.value));
        }

        // Category Filter Chips
        document.querySelectorAll('.chip-btn').forEach(chip => {
            chip.addEventListener('click', () => {
                document.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                this.filterDictionaryByCategory(chip.dataset.cat);
            });
        });

        // Video Upload Dropzone
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
                if (e.dataTransfer.files.length > 0) {
                    this.processUploadedVideo(e.dataTransfer.files[0]);
                }
            });
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.processUploadedVideo(e.target.files[0]);
                }
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

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
        
        const badgeStatus = document.getElementById('ws-status-badge');

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('[WS] Đã kết nối WebSocket thành công.');
            if (badgeStatus) {
                badgeStatus.innerHTML = '<span class="pulse-dot"></span> LIVE ENGINE';
                badgeStatus.className = 'badge badge-live';
            }
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onclose = () => {
            console.log('[WS] WebSocket đã đóng. Thử kết nối lại sau 2s...');
            if (badgeStatus) {
                badgeStatus.innerHTML = '⚠️ OFFLINE';
                badgeStatus.className = 'badge';
            }
            setTimeout(() => this.connectWebSocket(), 2000);
        };
    }

    async toggleCamera() {
        const placeholder = document.getElementById('viewport-placeholder');
        const btnToggleCam = document.getElementById('btn-toggle-cam');

        if (!this.isStreaming) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
                    audio: false
                });

                this.video.srcObject = stream;
                await this.video.play();

                this.canvas.width = this.video.videoWidth || 640;
                this.canvas.height = this.video.videoHeight || 360;

                this.isStreaming = true;
                placeholder.classList.add('hidden');
                btnToggleCam.innerHTML = '⏹️ Dừng Camera';
                btnToggleCam.classList.replace('btn-primary', 'btn-secondary');

                // Start transmission loop (~25 FPS)
                this.streamInterval = setInterval(() => this.sendFrame(), 40);
            } catch (err) {
                alert(`Không thể truy cập camera: ${err.message}`);
            }
        } else {
            if (this.video.srcObject) {
                this.video.srcObject.getTracks().forEach(track => track.stop());
            }
            clearInterval(this.streamInterval);
            this.isStreaming = false;
            placeholder.classList.remove('hidden');
            btnToggleCam.innerHTML = '🎥 Bật Camera';
            btnToggleCam.classList.replace('btn-secondary', 'btn-primary');
        }
    }

    sendFrame() {
        if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        const dataUrl = this.canvas.toDataURL('image/jpeg', 0.65);

        this.ws.send(JSON.stringify({ image: dataUrl }));

        // FPS Calculation
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

    handleWebSocketMessage(data) {
        if (data.type === 'prediction') {
            // 1. Update HUD Badge
            const predNameEl = document.getElementById('hud-pred-name');
            const predConfEl = document.getElementById('hud-pred-conf');
            const predIconEl = document.getElementById('hud-pred-icon');

            if (predNameEl) predNameEl.textContent = data.display;
            if (predConfEl) predConfEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;
            if (predIconEl) predIconEl.textContent = data.icon;

            // 2. Update Latency Telemetry
            const latEl = document.getElementById('tele-latency');
            if (latEl) latEl.textContent = `${data.latency_ms} ms`;

            // 3. Update Buffer Gauge
            const bufEl = document.getElementById('tele-buffer');
            if (bufEl) bufEl.textContent = `${data.buffer_size}/30`;

            // 4. Update Probability Bars
            this.renderProbabilityBars(data.top5);

            // 5. Update Sentence Builder (if confident & threshold met)
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
            const bufEl = document.getElementById('tele-buffer');
            if (bufEl) bufEl.textContent = `${data.buffer_size}/${data.required}`;
        }
    }

    renderProbabilityBars(top5) {
        const container = document.getElementById('prob-bars-container');
        if (!container || !top5) return;

        let html = '';
        top5.forEach((item, idx) => {
            const pct = (item.probability * 100).toFixed(1);
            const isTop = idx === 0;
            html += `
                <div class="prob-item ${isTop ? 'top' : ''}">
                    <div class="prob-meta">
                        <span class="prob-name">${item.icon} ${item.display}</span>
                        <span class="prob-val">${pct}%</span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
    }

    renderSentence() {
        const box = document.getElementById('sentence-box');
        if (!box) return;

        if (this.sentenceWords.length === 0) {
            box.innerHTML = '<span style="color: var(--text-muted); font-size: 0.95rem;">Chưa có từ nào được nhận diện. Hãy làm dấu trước camera...</span>';
        } else {
            box.innerHTML = this.sentenceWords.map(word => `<span class="word-chip">${word}</span>`).join('');
        }
    }

    initTTS() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = () => {
                this.voices = window.speechSynthesis.getVoices();
            };
        }
    }

    speakSentence() {
        const text = this.sentenceWords.join(' ');
        if (!text) return;

        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'vi-VN';
            utterance.rate = 0.95;

            // Try to pick a Vietnamese voice
            const viVoice = window.speechSynthesis.getVoices().find(v => v.lang.includes('vi') || v.name.includes('Vietnamese'));
            if (viVoice) utterance.voice = viVoice;

            window.speechSynthesis.speak(utterance);
        }
    }

    filterDictionary(query) {
        const term = query.toLowerCase().trim();
        document.querySelectorAll('.dict-card').forEach(card => {
            const name = card.dataset.name.toLowerCase();
            const desc = card.dataset.desc.toLowerCase();
            const cat = card.dataset.cat.toLowerCase();

            if (name.includes(term) || desc.includes(term) || cat.includes(term)) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });
    }

    filterDictionaryByCategory(category) {
        document.querySelectorAll('.dict-card').forEach(card => {
            if (!category || category === 'all' || card.dataset.cat === category) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });
    }

    async processUploadedVideo(file) {
        const progressEl = document.getElementById('upload-status');
        const resultsEl = document.getElementById('upload-results');
        const sentenceEl = document.getElementById('upload-sentence');
        const timelineBody = document.getElementById('upload-timeline-body');

        if (progressEl) progressEl.textContent = '⏳ Đang phân tích video qua mô hình VSL...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/predict_video', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();

            if (data.status === 'success') {
                if (progressEl) progressEl.textContent = `✅ Phân tích hoàn tất! (${data.total_frames} frames / ${data.duration_sec}s)`;
                if (resultsEl) resultsEl.style.display = 'block';
                if (sentenceEl) sentenceEl.textContent = data.sentence;

                if (timelineBody) {
                    timelineBody.innerHTML = data.timeline.map(item => `
                        <tr>
                            <td><span style="color: var(--accent-cyan); font-family: var(--font-mono);">${item.timestamp}</span></td>
                            <td><strong>${item.icon} ${item.display}</strong> (${item.name})</td>
                            <td><span class="badge badge-live">${item.confidence}%</span></td>
                        </tr>
                    `).join('');
                }
            } else {
                if (progressEl) progressEl.textContent = '❌ Không thể nhận diện video.';
            }
        } catch (err) {
            if (progressEl) progressEl.textContent = `❌ Lỗi tải lên: ${err.message}`;
        }
    }
}

// Instantiate on page load
document.addEventListener('DOMContentLoaded', () => {
    window.vslApp = new VSLApp();
});
