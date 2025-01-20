import * as fs from 'fs';
import * as path from 'path';

class Logger {
  private logFile: string;

  constructor(filename: string) {
    this.logFile = path.join(__dirname, '../../logs', filename);
    this.ensureLogDirectory();
  }

  private ensureLogDirectory(): void {
    const dir = path.dirname(this.logFile);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  public log(message: string): void {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${message}\n`;
    fs.appendFileSync(this.logFile, logMessage);
  }
}

export default Logger;
