/**
 * VSL Vision Studio — Application Client Engine
 * Handles WebSocket video/landmark streaming, TTS, Feedback modal, and UI Telemetry.
 */

class VSLApp {
    constructor() {
        this.ws = null;
        this.video = document.getElementById('video-element');
        this.canvas = document.getElementById('canvas-element');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        
        this.isStreaming = false;
        this.isMirror = true;
        this.streamInterval = null;
        this.fps = 0;
        this.frameCount = 0;
        this.lastFpsCalc = performance.now();

        this.sentenceWords = [];
        this.lastConfidentSign = null;
        this.lastSignTimestamp = 0;
        this.currentRawPred = '';

        this.initEventListeners();
        this.connectWebSocket();
        this.initTTS();
    }

    initEventListeners() {
        // Tab Navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
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
                if (this.video) this.video.style.transform = this.isMirror ? 'scaleX(-1)' : 'scaleX(1)';
                if (this.canvas) this.canvas.style.transform = this.isMirror ? 'scaleX(-1)' : 'scaleX(1)';
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
                    btnCopy.textContent = 'Đã chép!';
                    setTimeout(() => btnCopy.textContent = 'Sao chép', 2000);
                }
            });
        }

        // Feedback Modal Controls
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

        if (btnCloseModal && modal) {
            btnCloseModal.addEventListener('click', () => modal.classList.add('hidden'));
        }

        if (btnSubmitFeedback && modal) {
            btnSubmitFeedback.addEventListener('click', () => this.submitFeedback());
        }

        // Dictionary Search
        const dictSearch = document.getElementById('dict-search');
        if (dictSearch) {
            dictSearch.addEventListener('input', (e) => this.filterDictionary(e.target.value));
        }

        // Category Filter Chips
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
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
        const statusText = document.getElementById('ws-status-text');
        const statusDot = document.querySelector('.status-dot');

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            if (statusText) statusText.textContent = 'ONNX ENGINE';
            if (statusDot) statusDot.className = 'status-dot online';
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onclose = () => {
            if (statusText) statusText.textContent = 'OFFLINE';
            if (statusDot) statusDot.className = 'status-dot';
            setTimeout(() => this.connectWebSocket(), 2000);
        };
    }

    async toggleCamera() {
        const placeholder = document.getElementById('viewport-placeholder');
        const btnToggleCam = document.getElementById('btn-toggle-cam');
        const hudBadge = document.getElementById('hud-badge');

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
                if (placeholder) placeholder.classList.add('hidden');
                if (hudBadge) hudBadge.classList.remove('hidden');
                if (btnToggleCam) {
                    btnToggleCam.textContent = 'Dừng Camera';
                    btnToggleCam.classList.replace('btn-primary', 'btn-secondary');
                }

                this.streamInterval = setInterval(() => this.sendFrame(), 33);
            } catch (err) {
                alert(`Không thể truy cập camera: ${err.message}`);
            }
        } else {
            if (this.video && this.video.srcObject) {
                this.video.srcObject.getTracks().forEach(track => track.stop());
            }
            clearInterval(this.streamInterval);
            this.isStreaming = false;
            if (placeholder) placeholder.classList.remove('hidden');
            if (hudBadge) hudBadge.classList.add('hidden');
            if (btnToggleCam) {
                btnToggleCam.textContent = 'Bật Camera';
                btnToggleCam.classList.replace('btn-secondary', 'btn-primary');
            }
        }
    }

    sendFrame() {
        if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        const dataUrl = this.canvas.toDataURL('image/jpeg', 0.65);

        this.ws.send(JSON.stringify({ image: dataUrl }));

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
            this.currentRawPred = data.name;

            // HUD Badge
            const predNameEl = document.getElementById('hud-pred-name');
            const predConfEl = document.getElementById('hud-pred-conf');
            const predIconEl = document.getElementById('hud-pred-icon');
            const hudBadge = document.getElementById('hud-badge');

            if (predNameEl) predNameEl.textContent = data.display;
            if (predConfEl) predConfEl.textContent = `${(data.confidence * 100).toFixed(0)}%`;
            if (predIconEl) predIconEl.textContent = data.icon;

            if (hudBadge) {
                if (data.is_confident) {
                    hudBadge.classList.add('confident');
                } else {
                    hudBadge.classList.remove('confident');
                }
            }

            // Latency & Buffer
            const latEl = document.getElementById('tele-latency');
            if (latEl) latEl.textContent = `${data.latency_ms} ms`;

            const bufEl = document.getElementById('tele-buffer');
            if (bufEl) bufEl.textContent = `${data.buffer_size} / 30`;

            // Probability Bars
            this.renderProbabilityBars(data.top5);

            // Sentence Builder
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
            if (bufEl) bufEl.textContent = `${data.buffer_size} / ${data.required}`;
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
                <div class="prob-item">
                    <div class="prob-item-header">
                        <span class="prob-item-name ${isTop ? 'active' : ''}">${item.icon} ${item.display}</span>
                        <span class="prob-item-val ${isTop ? 'active' : ''}">${pct}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-bar ${isTop ? 'active' : ''}" style="width: ${pct}%"></div>
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
            box.innerHTML = '<span class="placeholder-text">Chưa ghi nhận cử chỉ nào. Hãy thực hiện ký hiệu trong khung hình...</span>';
        } else {
            box.innerHTML = this.sentenceWords.map(word => `<span class="sentence-word">${word}</span>`).join('');
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

            const viVoice = window.speechSynthesis.getVoices().find(v => v.lang.includes('vi') || v.name.includes('Vietnamese'));
            if (viVoice) utterance.voice = viVoice;

            window.speechSynthesis.speak(utterance);
        }
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
                    confidence: 0.85,
                    notes: notes
                })
            });
            const res = await resp.json();
            if (res.status === 'success') {
                alert('Cảm ơn bạn! Phản hồi đã được lưu vào hệ thống Active Learning.');
                if (modal) modal.classList.add('hidden');
            }
        } catch (e) {
            alert('Lỗi gửi phản hồi: ' + e.message);
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

        if (progressEl) {
            progressEl.classList.remove('hidden');
            progressEl.textContent = 'Đang phân tích video qua mô hình VSL...';
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/predict_video', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();

            if (data.status === 'success') {
                if (progressEl) progressEl.textContent = `Phân tích hoàn tất! (${data.total_frames} frames / ${data.duration_sec}s)`;
                if (resultsEl) resultsEl.classList.remove('hidden');
                if (sentenceEl) sentenceEl.textContent = data.sentence;

                if (timelineBody) {
                    timelineBody.innerHTML = data.timeline.map(item => `
                        <tr>
                            <td><code>${item.timestamp}</code></td>
                            <td><strong>${item.icon} ${item.display}</strong></td>
                            <td><span style="color: var(--accent-emerald); font-family: var(--font-mono);">${item.confidence}%</span></td>
                        </tr>
                    `).join('');
                }
            } else {
                if (progressEl) progressEl.textContent = 'Không thể nhận diện video.';
            }
        } catch (err) {
            if (progressEl) progressEl.textContent = `Lỗi tải lên: ${err.message}`;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.vslApp = new VSLApp();
});
