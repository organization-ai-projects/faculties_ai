# core/alert_system_web.py

from flask import Flask, render_template, jsonify
import os
import json
from core.alert_system import AlertSystem

app = Flask(__name__)

# Dossier des alertes
alert_system = AlertSystem()

@app.route('/')
def index():
    # Charger toutes les alertes depuis le dossier
    alerts = []
    alert_folder = alert_system.alert_folder
    for alert_file in os.listdir(alert_folder):
        if alert_file.endswith('.json'):
            with open(os.path.join(alert_folder, alert_file), 'r') as f:
                alert_data = json.load(f)
                alerts.append(alert_data)
    return render_template('index.html', alerts=alerts)

@app.route('/alert/<alert_id>', methods=['GET'])
def alert_detail(alert_id):
    alert_file = os.path.join(alert_system.alert_folder, f'{alert_id}.json')
    if os.path.exists(alert_file):
        with open(alert_file, 'r') as f:
            alert_data = json.load(f)
            return render_template('alert_detail.html', alert=alert_data)
    return "Alerte non trouvée", 404

@app.route('/resolve_alert/<alert_id>', methods=['POST'])
def resolve_alert(alert_id):
    alert_file = os.path.join(alert_system.alert_folder, f'{alert_id}.json')
    if os.path.exists(alert_file):
        with open(alert_file, 'r') as f:
            alert_data = json.load(f)
            alert_data['status'] = 'resolved'
            with open(alert_file, 'w') as f_update:
                json.dump(alert_data, f_update, indent=4)
            return jsonify({"message": "Alerte résolue avec succès!"}), 200
    return "Alerte non trouvée", 404

if __name__ == '__main__':
    app.run(debug=True)
