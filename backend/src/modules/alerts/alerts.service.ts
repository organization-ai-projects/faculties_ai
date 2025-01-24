import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class AlertsService {
  private readonly alertFolder = path.resolve('ai-services/app/alerts/alerts');

  getAllAlerts() {
    const alerts = [];
    const files = fs.readdirSync(this.alertFolder);
    for (const file of files) {
      if (file.endsWith('.json')) {
        const filePath = path.join(this.alertFolder, file);
        const alertData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        alerts.push(alertData);
      }
    }
    return alerts;
  }

  getAlert(id: string) {
    const filePath = path.join(this.alertFolder, `${id}.json`);
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    }
    throw new Error('Alerte non trouvée');
  }

  resolveAlert(id: string) {
    const filePath = path.join(this.alertFolder, `${id}.json`);
    if (fs.existsSync(filePath)) {
      const alertData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      alertData.status = 'resolved';
      fs.writeFileSync(filePath, JSON.stringify(alertData, null, 4));
      return { message: 'Alerte résolue avec succès!' };
    }
    throw new Error('Alerte non trouvée');
  }
}
