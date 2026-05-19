// 全局状态
const state = {
    uploadedImages: [],
    isGenerating: false,
    chatHistory: [],
    currentChatId: null,
    theme: localStorage.getItem('theme') || 'light',
    modalImages: [],
    currentModalIndex: 0
};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initDragAndDrop();
    loadChatHistory();
    updateUI();
    initModalKeyboard();
});

// ==================== 主题切换 ====================
function initTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
}

function toggleTheme() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('theme', state.theme);
}

// ==================== 侧边栏 ====================
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('collapsed');
}

function startNewChat() {
    state.currentChatId = null;
    state.uploadedImages = [];
    document.getElementById('messagesList').innerHTML = '';
    document.getElementById('welcomeScreen').style.display = 'flex';
    document.getElementById('messageInput').value = '';
    renderUploadedImages();
    updateUI();
    
    const chatId = Date.now();
    const chatTitle = '新对话 ' + new Date().toLocaleTimeString();
    state.chatHistory.unshift({ id: chatId, title: chatTitle });
    renderChatHistory();
}

function renderChatHistory() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = state.chatHistory.map(chat => `
        <div class="history-item" onclick="loadChat(${chat.id})">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
            </svg>
            <span>${chat.title}</span>
        </div>
    `).join('');
}

function loadChatHistory() {
    state.chatHistory = [
        { id: 1, title: '风景画创作' },
        { id: 2, title: '人物肖像融合' }
    ];
    renderChatHistory();
}

// ==================== 图片上传 ====================
function triggerUpload() {
    document.getElementById('fileInput').click();
}

function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;
    
    files.forEach(file => {
        if (!file.type.startsWith('image/')) {
            alert('请上传图片文件');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) {
            alert('图片大小不能超过10MB');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            state.uploadedImages.push(e.target.result);
            renderUploadedImages();
            updateUI();
        };
        reader.readAsDataURL(file);
    });
    
    event.target.value = '';
}

function renderUploadedImages() {
    const container = document.getElementById('uploadSlots');
    const addBtn = document.getElementById('uploadAddBtn');
    
    // 清除所有已存在的图片项（保留添加按钮）
    const existingItems = container.querySelectorAll('.uploaded-image-item');
    existingItems.forEach(item => item.remove());
    
    // 按顺序重新渲染所有图片
    state.uploadedImages.forEach((src, index) => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'uploaded-image-item';
        itemDiv.dataset.index = index;
        itemDiv.innerHTML = `
            <img src="${src}" alt="上传图片${index + 1}">
            <button class="remove-btn" onclick="removeImage(${index})">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        `;
        container.insertBefore(itemDiv, addBtn);
    });
    
    updateUI();
}

function removeImage(index) {
    state.uploadedImages.splice(index, 1);
    renderUploadedImages();
}

function initDragAndDrop() {
    const addBtn = document.getElementById('uploadAddBtn');
    
    addBtn.addEventListener('dragover', (e) => {
        e.preventDefault();
        addBtn.style.borderColor = 'var(--primary-color)';
        addBtn.style.background = 'rgba(102, 126, 234, 0.1)';
    });
    
    addBtn.addEventListener('dragleave', () => {
        addBtn.style.borderColor = '';
        addBtn.style.background = '';
    });
    
    addBtn.addEventListener('drop', (e) => {
        e.preventDefault();
        addBtn.style.borderColor = '';
        addBtn.style.background = '';
        
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        files.forEach(file => {
            if (file.size <= 10 * 1024 * 1024) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    state.uploadedImages.push(event.target.result);
                    renderUploadedImages();
                    updateUI();
                };
                reader.readAsDataURL(file);
            }
        });
    });
}

// ==================== 消息发送 ====================
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function clearAll() {
    state.uploadedImages = [];
    const container = document.getElementById('uploadSlots');
    const items = container.querySelectorAll('.uploaded-image-item');
    items.forEach(item => item.remove());
    document.getElementById('messageInput').value = '';
    updateUI();
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message && state.uploadedImages.length === 0) return;
    if (state.isGenerating) return;
    
    document.getElementById('welcomeScreen').style.display = 'none';
    
    // 添加用户消息
    const userMessage = {
        role: 'user',
        content: message,
        images: [...state.uploadedImages]
    };
    
    addMessageToChat(userMessage);
    
    // 清空输入
    input.value = '';
    input.style.height = 'auto';
    
    const uploadedImageData = [...state.uploadedImages];
    
    // 清空上传区
    state.uploadedImages = [];
    const container = document.getElementById('uploadSlots');
    const items = container.querySelectorAll('.uploaded-image-item');
    items.forEach(item => item.remove());
    
    // 开始AI生成
    await generateWithSyncAnimation(uploadedImageData);
}

function addMessageToChat(message) {
    const messagesList = document.getElementById('messagesList');
    const messageEl = document.createElement('div');
    messageEl.className = `message ${message.role}`;
    
    const avatar = message.role === 'user' ? 'U' : 'AI';
    const imagesHtml = message.images && message.images.length > 0 
        ? `<div class="message-images">${message.images.map((src, i) => `
            <div class="message-image" onclick="openImageModal([${message.images.map(s => `'${s}'`).join(',')}], ${i})">
                <img src="${src}" alt="上传图片${i + 1}" class="clickable-image">
            </div>
        `).join('')}</div>`
        : '';
    
    const contentHtml = message.content 
        ? `<div class="message-text">${escapeHtml(message.content)}</div>`
        : '';
    
    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${imagesHtml}
            ${contentHtml}
        </div>
    `;
    
    messagesList.appendChild(messageEl);
    scrollToBottom();
}

// ==================== 同步生成动画 ====================
async function generateWithSyncAnimation(uploadedImages) {
    state.isGenerating = true;
    updateUI();
    
    const messagesList = document.getElementById('messagesList');
    const messageEl = document.createElement('div');
    messageEl.className = 'message assistant';
    
    // 生成唯一ID用于这次生成
    const generationId = 'gen_' + Date.now();
    
    // 创建消息结构
    messageEl.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="thinking-row" id="thinkingRow_${generationId}">
                <svg class="thinking-spinner" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"></path>
                </svg>
                <span class="thinking-text">正在分析图片特征并生成图像...</span>
            </div>
            <div class="generation-result" id="resultContainer_${generationId}">
                <div class="result-header">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                    <span>生成结果</span>
                </div>
                <div class="result-image-container">
                    <div class="result-image-wrapper" id="imageWrapper_${generationId}">
                        <div class="generation-mask" id="generationMask_${generationId}"></div>
                        <img src="" alt="generated" id="resultImage_${generationId}" style="cursor: zoom-in;">
                    </div>
                </div>
                <div class="result-actions">
                    <button class="result-btn" onclick="downloadResultImage('${generationId}')">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        下载
                    </button>
                    <button class="result-btn" onclick="regenerate('${generationId}')">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="23 4 23 10 17 10"></polyline>
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                        </svg>
                        重新生成
                    </button>
                    <button class="result-btn" onclick="shareResult()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="18" cy="5" r="3"></circle>
                            <circle cx="6" cy="12" r="3"></circle>
                            <circle cx="18" cy="19" r="3"></circle>
                            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                        </svg>
                        分享
                    </button>
                </div>
            </div>
        </div>
    `;
    
    messagesList.appendChild(messageEl);
    scrollToBottom();
    
    const thinkingRow = messageEl.querySelector(`#thinkingRow_${generationId}`);
    const generationMask = messageEl.querySelector(`#generationMask_${generationId}`);
    const resultImage = messageEl.querySelector(`#resultImage_${generationId}`);
    
    // 生成图片
    const generatedImage = createDefaultImage();
    resultImage.src = generatedImage;
    resultImage.dataset.src = generatedImage;
    
    // 绑定点击放大事件
    resultImage.onclick = () => openImageModal([generatedImage], 0);
    
    // 创建分段遮罩（从上到下显示）
    const segments = 12; // 分成12段
    const segmentHeight = 100 / segments;
    
    for (let i = 0; i < segments; i++) {
        const segment = document.createElement('div');
        segment.className = 'generation-segment';
        segment.style.top = `${i * segmentHeight}%`;
        segment.style.height = `${segmentHeight}%`;
        segment.id = `segment_${generationId}_${i}`;
        generationMask.appendChild(segment);
    }
    
    // 第一阶段：从上到下逐段显示（带随机卡顿）
    for (let i = 0; i < segments; i++) {
        const segment = messageEl.querySelector(`#segment_${generationId}_${i}`);
        if (segment) {
            // 随机延迟 200ms - 800ms，产生卡顿感
            const delay = 200 + Math.random() * 600;
            await new Promise(resolve => setTimeout(resolve, delay));
            segment.style.opacity = '0';
        }
        scrollToBottom();
    }
    
    // 第二阶段：清晰化（4秒）
    resultImage.classList.add('sharpening');
    await new Promise(resolve => setTimeout(resolve, 4000));
    
    // 完成状态
    thinkingRow.classList.add('completed');
    thinkingRow.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
        <span class="thinking-text">生成完成</span>
    `;
    
    // 移除遮罩和动画类
    generationMask.style.opacity = '0';
    resultImage.classList.remove('sharpening');
    resultImage.style.filter = 'blur(0) brightness(1)';
    resultImage.style.opacity = '1';
    
    state.isGenerating = false;
    updateUI();
    scrollToBottom();
}

function createDefaultImage() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    
    const gradient = ctx.createLinearGradient(0, 0, 512, 512);
    gradient.addColorStop(0, '#667eea');
    gradient.addColorStop(0.5, '#764ba2');
    gradient.addColorStop(1, '#f093fb');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 512, 512);
    
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 2;
    for (let i = 0; i < 5; i++) {
        ctx.beginPath();
        ctx.arc(256, 256, 50 + i * 40, 0, Math.PI * 2);
        ctx.stroke();
    }
    
    ctx.fillStyle = 'white';
    ctx.font = 'bold 32px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('AI Generated', 256, 256);
    ctx.font = '18px Inter, sans-serif';
    ctx.fillText('艺术作品', 256, 290);
    
    return canvas.toDataURL();
}

// ==================== 图片放大模态框 ====================
function openImageModal(images, index) {
    state.modalImages = images;
    state.currentModalIndex = index;
    
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    
    modalImage.src = images[index];
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    
    updateModalNav();
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
    state.modalImages = [];
    state.currentModalIndex = 0;
}

function changeImage(direction) {
    const newIndex = state.currentModalIndex + direction;
    if (newIndex >= 0 && newIndex < state.modalImages.length) {
        state.currentModalIndex = newIndex;
        document.getElementById('modalImage').src = state.modalImages[newIndex];
        updateModalNav();
    }
}

function updateModalNav() {
    const prevBtn = document.getElementById('modalPrev');
    const nextBtn = document.getElementById('modalNext');
    
    prevBtn.classList.toggle('hidden', state.currentModalIndex === 0);
    nextBtn.classList.toggle('hidden', state.currentModalIndex === state.modalImages.length - 1);
}

function downloadModalImage() {
    const src = document.getElementById('modalImage').src;
    downloadImage(src);
}

function downloadResultImage(generationId) {
    const resultImage = document.getElementById(`resultImage_${generationId}`);
    if (resultImage && resultImage.dataset.src) {
        downloadImage(resultImage.dataset.src);
    }
}

function downloadImage(src) {
    const link = document.createElement('a');
    link.href = src;
    link.download = 'ai-generated-image-' + Date.now() + '.png';
    link.click();
}

function openInNewTab() {
    const src = document.getElementById('modalImage').src;
    window.open(src, '_blank');
}

function initModalKeyboard() {
    document.addEventListener('keydown', (e) => {
        const modal = document.getElementById('imageModal');
        if (!modal.classList.contains('active')) return;
        
        if (e.key === 'Escape') {
            closeImageModal();
        } else if (e.key === 'ArrowLeft') {
            changeImage(-1);
        } else if (e.key === 'ArrowRight') {
            changeImage(1);
        }
    });
}

// ==================== 结果操作 ====================
function regenerate() {
    if (state.isGenerating) return;
    
    const imageWrapper = document.getElementById('imageWrapper');
    const resultImage = document.getElementById('resultImage');
    const thinkingRow = document.getElementById('thinkingRow');
    
    if (imageWrapper && resultImage) {
        // 重置思考状态
        if (thinkingRow) {
            thinkingRow.classList.remove('completed');
            thinkingRow.innerHTML = `
                <svg class="thinking-spinner" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"></path>
                </svg>
                <span class="thinking-text">正在重新生成...</span>
            `;
        }
        
        // 重新添加动画类
        imageWrapper.classList.remove('generating');
        void imageWrapper.offsetWidth; // 触发重排
        imageWrapper.classList.add('generating');
        
        // 新图片
        setTimeout(() => {
            const newImage = createDefaultImage();
            resultImage.src = newImage;
            resultImage.dataset.src = newImage;
        }, 100);
        
        // 6秒后完成
        setTimeout(() => {
            if (thinkingRow) {
                thinkingRow.classList.add('completed');
                thinkingRow.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                    <span class="thinking-text">生成完成</span>
                `;
            }
            imageWrapper.classList.remove('generating');
        }, 6000);
    }
}

function shareResult() {
    if (navigator.share) {
        navigator.share({
            title: 'AI生成的艺术作品',
            text: '看看我用AI创作的艺术作品！',
            url: window.location.href
        }).catch(() => {
            copyToClipboard();
        });
    } else {
        copyToClipboard();
    }
}

function copyToClipboard() {
    const textarea = document.createElement('textarea');
    textarea.value = window.location.href;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    
    showToast('链接已复制到剪贴板');
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--text-primary);
        color: var(--bg-primary);
        padding: 12px 24px;
        border-radius: 24px;
        font-size: 14px;
        z-index: 2000;
        animation: fadeIn 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

// ==================== 工具函数 ====================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    container.scrollTop = container.scrollHeight;
}

function updateUI() {
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = state.isGenerating;
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateX(-50%) translateY(10px); }
        to { opacity: 1; transform: translateX(-50%) translateY(0); }
    }
    @keyframes fadeOut {
        from { opacity: 1; transform: translateX(-50%) translateY(0); }
        to { opacity: 0; transform: translateX(-50%) translateY(10px); }
    }
`;
document.head.appendChild(style);