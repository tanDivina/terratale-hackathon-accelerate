document.addEventListener('DOMContentLoaded', () => {
    // --- HTML Elements ---
    const recordBtn = document.getElementById('record-btn');
    const textInput = document.getElementById('text-input');
    const sendBtn = document.getElementById('send-btn');
    const statusText = document.getElementById('status-text');
    const mateoAudio = document.getElementById('response-audio');
    const textResponseArea = document.getElementById('text-response-area');
    const imageResultsArea = document.getElementById('image-results-area');
    const playMateoBtn = document.getElementById('play-btn');
    const readPapitoBtn = document.getElementById('read-aloud-btn');
    let papitoAudio = new Audio();

    // --- WebSocket Logic ---
    function connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        socket = new WebSocket(`${wsProtocol}${window.location.host}/ws`);

        socket.onopen = () => {
            statusText.textContent = 'Ask a question with your voice or text.';
            [recordBtn, sendBtn].forEach(btn => btn.disabled = false);
        };

        socket.onmessage = (event) => {
            if (event.data instanceof Blob) {
                audioChunks.push(event.data);
                return;
            }
            try {
                const message = JSON.parse(event.data);
                switch (message.type) {
                    case 'text':
                        textResponseArea.innerText = message.content;
                        textResponseArea.style.display = 'block';
                        readPapitoBtn.style.display = 'inline-block'; // Make button visible
                        readPapitoBtn.disabled = false;
                        break;
                    case 'image_search_results':
                        displayImages(message.content);
                        break;
                    case 'audio_end':
                        if (audioChunks.length > 0) prepareAudioForPlayback();
                        break;
                    case 'error':
                        statusText.textContent = `Server error: ${message.content}`;
                        break;
                }
            } catch (e) { console.error("Failed to parse JSON:", e); }
        };

        socket.onclose = () => {
            statusText.textContent = 'Connection lost. Please refresh.';
            [recordBtn, sendBtn, playMateoBtn, readPapitoBtn].forEach(btn => btn.disabled = true);
        };
    }

    // --- Main Query Handler ---
    function sendQuery(queryText) {
        if (!queryText) return;
        // Reset UI
        [textResponseArea, mateoAudio, imageResultsArea].forEach(el => el.style.display = 'none');
        [playMateoBtn, readPapitoBtn].forEach(btn => { btn.disabled = true; btn.style.display = 'none'; });
        textResponseArea.innerText = '';
        imageResultsArea.innerHTML = '';
        statusText.textContent = `Processing: "${queryText}"`;

        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(queryText);
        } else {
            connectWebSocket();
            statusText.textContent = 'Reconnecting... Please try again.';
        }
    }

    // --- Event Listeners ---
    sendBtn.addEventListener('click', () => sendQuery(textInput.value));
    textInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') sendQuery(textInput.value); });
    playMateoBtn.addEventListener('click', () => mateoAudio.play());
    readPapitoBtn.addEventListener('click', async () => {
        const textToRead = textResponseArea.innerText;
        if (!textToRead) return;
        try {
            readPapitoBtn.disabled = true;
            readPapitoBtn.textContent = 'Synthesizing...';
            const response = await fetch('/synthesize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textToRead })
            });
            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            papitoAudio.src = audioUrl;
            papitoAudio.play();
            readPapitoBtn.textContent = 'Playing Papito...';
        } catch (error) {
            console.error("Synthesis failed:", error);
            readPapitoBtn.textContent = 'Read Aloud';
            readPapitoBtn.disabled = false;
        }
    });

    papitoAudio.onended = () => {
        readPapitoBtn.textContent = 'Read Aloud';
        readPapitoBtn.disabled = false;
    };

    // --- Mateo's Audio Playback ---
    async function prepareAudioForPlayback() {
        const rawAudioData = await (new Blob(audioChunks)).arrayBuffer();
        audioChunks = [];
        const waveFileBuffer = createWaveFile(rawAudioData);
        const audioBlob = new Blob([waveFileBuffer], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        mateoAudio.src = audioUrl;
        mateoAudio.style.display = 'block';
        playMateoBtn.disabled = false;
        statusText.textContent = 'Audio summary is ready.';
    }

    mateoAudio.onplay = () => statusText.textContent = "Playing Mateo's summary...";
    mateoAudio.onended = () => statusText.textContent = 'Ask a question with your voice or text.';

    function displayImages(images) {
        imageResultsArea.innerHTML = ''; // Clear previous results
        if (images.length > 0) {
            const identifiedName = images[0].fields.label ? images[0].fields.label[0] : "an unknown animal or plant";
            const identificationDiv = document.createElement('div');
            identificationDiv.innerHTML = `I think you saw a <strong>${identifiedName}</strong>! Here are some photos:`;
            imageResultsArea.appendChild(identificationDiv);
        }
        images.forEach(image => {
            const imgElement = document.createElement('img');
            imgElement.src = image.fields.photo_image_url[0];
            imgElement.alt = image.fields.photo_description[0];
            imageResultsArea.appendChild(imgElement);
        });
        imageResultsArea.style.display = 'block';
    }

    // --- Speech Recognition ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { recordBtn.disabled = true; } else {
        const recognition = new SpeechRecognition();
        recognition.interimResults = false;
        recordBtn.addEventListener('click', () => {
            recordBtn.classList.contains('recording') ? recognition.stop() : recognition.start();
        });
        recognition.onstart = () => recordBtn.classList.add('recording');
        recognition.onend = () => recordBtn.classList.remove('recording');
        recognition.onresult = (event) => sendQuery(event.results[0][0].transcript);
    }
    
    // --- WAV File Creation ---
    function createWaveFile(audioData) {
        const sampleRate = 24000, numChannels = 1, bitsPerSample = 16, dataSize = audioData.byteLength, blockAlign = (numChannels * bitsPerSample) / 8, byteRate = sampleRate * blockAlign, buffer = new ArrayBuffer(44 + dataSize), view = new DataView(buffer);
        function writeString(view, offset, string) { for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i)); }
        writeString(view, 0, 'RIFF'); view.setUint32(4, 36 + dataSize, true); writeString(view, 8, 'WAVE'); writeString(view, 12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, numChannels, true); view.setUint32(24, sampleRate, true); view.setUint32(28, byteRate, true); view.setUint16(32, blockAlign, true); view.setUint16(34, bitsPerSample, true); writeString(view, 36, 'data'); view.setUint32(40, dataSize, true); new Uint8Array(buffer, 44).set(new Uint8Array(audioData));
        return buffer;
    }

    connectWebSocket();
});

// --- Style Enhancements ---
const style = document.createElement('style');
style.textContent = `
    .recording {
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    #response-audio, #text-response-area, #read-aloud-btn, #image-results-area {
        display: none; /* Initially hidden */
    }
    #image-results-area img {
        max-width: 200px;
        margin: 5px;
        border-radius: 5px;
    }
`;
document.head.appendChild(style);