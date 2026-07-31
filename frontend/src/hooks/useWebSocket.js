import { useEffect, useState } from 'react';
import { wsService } from '../services/websocket';

export const useWebSocket = () => {
  const [status, setStatus] = useState('disconnected');
  const [lastMessage, setLastMessage] = useState(null);

  useEffect(() => {
    wsService.connect();

    const unsubscribe = wsService.subscribe((msg) => {
      if (msg.type === 'STATUS') {
        setStatus(msg.status);
      } else {
        setLastMessage(msg);
      }
    });

    return () => {
      unsubscribe();
    };
  }, []);

  const sendMessage = (data) => wsService.send(data);

  return { status, lastMessage, sendMessage };
};
