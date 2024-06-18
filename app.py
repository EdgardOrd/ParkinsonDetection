from flask import Flask, request, render_template, url_for
import pandas as pd
import os

app = Flask(__name__)

# Ruta al archivo TSV con la información de los pacientes
patients_info_tsv = os.path.join(app.root_path, 'participants.tsv')

# Leer el archivo TSV con la información de los pacientes
patients_info = pd.read_csv(patients_info_tsv, delimiter='\t')

@app.route('/', methods=['GET', 'POST'])
def index():
    patient_id = None
    patient_info = None
    spectrogram_path = None
    bdi_path = None
    diagnosis_message = None

    if request.method == 'POST':
        patient_id = request.form.get('patient_id')

        # Buscar la información del paciente
        patient_info = patients_info[patients_info['participant_id'] == patient_id].to_dict(orient='records')
        if patient_info:
            patient_info = patient_info[0]

            # Definir el mensaje de diagnóstico basado en el valor de GROUP
            group = patient_info['GROUP']
            if group == 'PD':
                diagnosis_message = "El paciente tiene Parkinson Disease (PD)."
            elif group == 'PDD':
                diagnosis_message = "El paciente tiene Parkinson's Disease Dementia (PDD)."
            elif group == 'PD-MCI':
                diagnosis_message = "El paciente tiene Parkinson's Disease with Mild Cognitive Impairment (PD-MCI)."
            elif group == 'Control':
                diagnosis_message = "El paciente es un control."
            else:
                diagnosis_message = "Diagnóstico desconocido."

            # Buscar la imagen del espectrograma
            spectrogram_filename = f"{patient_id}-epo.png"
            spectrogram_path = url_for('static', filename=f"espectrogramas/{spectrogram_filename}")

            # Buscar la imagen de la banda de frecuencia
            bdi_filename = f"{patient_id}-epo_topomap.png"
            bdi_path = url_for('static', filename=f"bandas/{bdi_filename}")

            # Verificar que ambos archivos existan
            if not os.path.exists(os.path.join(app.root_path, 'static', 'espectrogramas', spectrogram_filename)):
                spectrogram_path = None
            if not os.path.exists(os.path.join(app.root_path, 'static', 'bandas', bdi_filename)):
                bdi_path = None

    return render_template('index.html', patient_id=patient_id, patient_info=patient_info, spectrogram_path=spectrogram_path, bdi_path=bdi_path, diagnosis_message=diagnosis_message)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
