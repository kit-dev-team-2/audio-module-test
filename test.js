const WS_URL = 'ws://localhost:8080';
const ws = new WebSocket(WS_URL);

ws.onopen = () => {
    console.log('WS connected');

    const msg = {
        type: 'config_update',
        config: {
            CONF_THRESH: 0.3,
        },
    };

    ws.send(JSON.stringify(msg));
    console.log('config_update sent');
};

ws.onmessage = (event) => {
    let payload;
    try {
        payload = JSON.parse(event.data);
    } catch {
        console.log('RX raw:', event.data);
        return;
    }

    if (payload.type === 'ping') {
        // 🔹 Node가 보낸 ping 에 대해 pong 응답
        ws.send(JSON.stringify({ type: 'pong', t: payload.t }));
        return;
    }

    console.log('RX:', payload);
};
