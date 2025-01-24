import React, { useEffect, useState } from 'react';
import axios from 'axios';

const AlertsPage = () => {
  interface Alert {
    alert_id: number;
    teacher: string;
    status: string;
  }

  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    axios.get('/api/alerts').then((response) => {
      setAlerts(response.data);
    });
  }, []);

  return (
    <div>
      <h1>Liste des alertes</h1>
      <ul>
        {alerts.map((alert) => (
          <li key={alert.alert_id}>
            <a href={`/alerts/${alert.alert_id}`}>
              {alert.teacher} - {alert.status}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AlertsPage;
