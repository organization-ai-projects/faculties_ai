// core/utils/logger.ts
import * as fs from 'fs';

export enum LogLevel {
    DEBUG = 'DEBUG',
    INFO = 'INFO',
    WARN = 'WARN',
    ERROR = 'ERROR',
}

export class Logger {
    private static logLevel: LogLevel = LogLevel.INFO;
    private static logFile: string | null = null;

    /**
     * Configure le logger avec un niveau et une sortie fichier optionnelle.
     * @param level - Niveau de log minimum.
     * @param filePath - Fichier où écrire les logs (facultatif).
     */
    static configure(level: LogLevel, filePath?: string): void {
        this.logLevel = level;
        if (filePath) {
            this.logFile = filePath;
            fs.writeFileSync(filePath, '', { flag: 'w' }); // Réinitialiser le fichier
        }
    }

    static debug(message: string): void {
        this.log(LogLevel.DEBUG, message);
    }

    static info(message: string): void {
        this.log(LogLevel.INFO, message);
    }

    static warn(message: string): void {
        this.log(LogLevel.WARN, message);
    }

    static error(message: string): void {
        this.log(LogLevel.ERROR, message);
    }

    private static log(level: LogLevel, message: string): void {
        if (this.shouldLog(level)) {
            const logMessage = `[${new Date().toISOString()}] [${level}]: ${message}`;
            
            // Afficher dans la console
            console.log(logMessage);

            // Écrire dans le fichier si configuré
            if (this.logFile) {
                fs.appendFileSync(this.logFile, logMessage + '\n');
            }
        }
    }

    private static shouldLog(level: LogLevel): boolean {
        const levels = Object.values(LogLevel);
        return levels.indexOf(level) >= levels.indexOf(this.logLevel);
    }
}
