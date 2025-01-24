import { Controller, Get, Param, Post } from '@nestjs/common';
import { AlertsService } from './alerts.service';

@Controller('alerts')
export class AlertsController {
  constructor(private readonly alertsService: AlertsService) {}

  @Get()
  getAllAlerts() {
    return this.alertsService.getAllAlerts();
  }

  @Get(':id')
  getAlert(@Param('id') id: string) {
    return this.alertsService.getAlert(id);
  }

  @Post(':id/resolve')
  resolveAlert(@Param('id') id: string) {
    return this.alertsService.resolveAlert(id);
  }
}
