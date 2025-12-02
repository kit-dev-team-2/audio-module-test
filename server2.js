// server.js
const express = require('express');
const http = require('http');
const WebSocket = require('ws');

const app = express();
app.get('/', (_, res) => res.send('WS server OK'));
const server = http.createServer(app);

const wss = new WebSocket.Server({ server });

// 포트 / 호스트를 환경변수에서 읽도록 (없으면 기본값)
const PORT = process.env.PORT || 8080;
const HOST = process.env.HOST || '0.0.0.0';

wss.on('connection', (ws, req) => {
    const ip = req.socket.remoteAddress;
    console.log('------------------------------');
    console.log('WS connected:', ip);
    console.log('------------------------------');

    ws.on('message', (data, isBinary) => {
        if (!isBinary) {
            const msg = data.toString();
            let payload = null;

            try {
                payload = JSON.parse(msg);
            } catch (e) {}

            if (payload && payload.type === 'detection') {
                console.log('RX detection:', payload);

                wss.clients.forEach((client) => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify(payload));
                    }
                });

                return;
            }

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

// 여기만 수정
server.listen(PORT, HOST, () => {
    console.log(`HTTP/WS on http://${HOST}:${PORT}`);
});
