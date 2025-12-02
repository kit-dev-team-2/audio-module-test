// server.js
const express = require('express');
const http = require('http');
const WebSocket = require('ws');

const app = express();
app.get('/', (_, res) => res.send('WS server OK'));
const server = http.createServer(app);

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
    const ip = req.socket.remoteAddress;
    console.log('------------------------------');
    console.log('WS connected:', ip);
    console.log('------------------------------');

    ws.on('message', (data, isBinary) => {
        if (!isBinary) {
            const msg = data.toString();
            let payload = null;

            // JSON 시도
            try {
                payload = JSON.parse(msg);
            } catch (e) {
                // JSON 아니면 그냥 문자열로 취급
            }

            // 🔹 파이썬에서 오는 detection 메시지면 → 전체 브로드캐스트
            if (payload && payload.type === 'detection') {
                console.log('RX detection:', payload);

                // 모든 클라이언트에게 그대로 전달
                wss.clients.forEach((client) => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify(payload));
                    }
                });

                return; // 여기서 처리 끝
            }

            // 🔹 그 외(메타퀘스트/브라우저에서 보낸 일반 텍스트)는 echo
            console.log('RX:', msg);
            ws.send(JSON.stringify({ type: 'ack', t: Date.now(), echo: msg }));
        } else {
            console.log('RX bin:', data.length, 'bytes');
            ws.send(JSON.stringify({ type: 'ack-bin', bytes: data.length }));
        }
    });

    ws.on('close', () => console.log('WS closed', ip));

    const iv = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping', t: Date.now() }));
        else clearInterval(iv);
    }, 3000);
});

server.listen(8080, '0.0.0.0', () => {
    console.log('HTTP/WS on http://0.0.0.0:8080');
});
