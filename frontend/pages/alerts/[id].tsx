import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';

interface Alert {
  teacher: string;
  status: string;
  timestamp: number;
  responses: string[];
}

const AlertDetailPage = () => {
  const router = useRouter();
  const { id } = router.query;
  const [alert, setAlert] = useState<Alert | null>(null);

  useEffect(() => {
    if (id) {
      axios.get(`/api/alerts/${id}`).then((response) => {
        setAlert(response.data);
      });
    }
  }, [id]);

  if (!alert) return <div>Chargement...</div>;

  return (
    <div>
      <h1>Détails de l'alerte</h1>
      <p>Professeur : {alert.teacher}</p>
      <p>Status : {alert.status}</p>
      <p>Timestamp : {new Date(alert.timestamp * 1000).toLocaleString()}</p>
      <p>Réponses incohérentes : {alert.responses.join(', ')}</p>
    </div>
  );
};

export default AlertDetailPage;
