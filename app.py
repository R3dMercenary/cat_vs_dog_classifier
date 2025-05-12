from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import os

app = Flask(__name__)

# Carga de modelos
try:
    modelo1 = load_model('static/models/cnn.h5')
    print("Modelo 1 cargado exitosamente")
except Exception as e:
    print(f"Error cargando modelo cnn: {e}")
    modelo1 = None

try:
    modelo2 = load_model('static/models/normal.h5')
    print("Modelo 2 cargado exitosamente")
except Exception as e:
    print(f"Error cargando modelo normal: {e}")
    modelo2 = None

# Configuración de cámara
camara = cv2.VideoCapture(0)

def preprocesar_imagen(img):
    """Convertir imágenes a escala de grises: (100,100,1)"""
    img = Image.fromarray(img).convert('L')
    img = img.resize((100, 100))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

def predecir_imagen(img_array, modelo):
    """Generar predicciones del modelo especificado"""
    if modelo is None:
        return ('Modelo no cargado', 0.5)
    
    try:
        predicciones = modelo.predict(img_array)
        
        if predicciones.shape[-1] == 1:
            if predicciones[0][0] > 0.5:
                return ('Perro', float(predicciones[0][0]))
            else:
                return ('Gato', 1 - float(predicciones[0][0]))
        elif predicciones.shape[-1] == 2:
            if predicciones[0][0] > predicciones[0][1]:
                return ('Gato', float(predicciones[0][0]))
            else:
                return ('Perro', float(predicciones[0][1]))
        else:
            return ('Salida inesperada', 0.5)
    except Exception as e:
        print(f"Error de predicción: {e}")
        return ('Error de predicción', 0.5)

def generar_frames():
    """Generar frames de la cámara"""
    while True:
        exito, frame = camara.read()
        if not exito:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def inicio():
    """Página principal con transmisión de video"""
    # Imágenes de la universidad
    imagenes_universidad = [
        {'path': 'static/images/unison.png', 'caption': 'Logo Universidad'},
        {'path': 'static/images/mcd.png', 'caption': 'Maestría Ciencia de Datos'}
    ]
    return render_template('index.html', imagenes_universidad=imagenes_universidad)

@app.route('/video_feed')
def video_feed():
    """Ruta de transmisión de video"""
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capturar', methods=['POST'])
def capturar():
    """Capturar imagen y generar predicciones"""
    exito, frame = camara.read()
    if not exito:
        return jsonify({'error': 'Error al capturar imagen'}), 400
    
    try:
        img_array = preprocesar_imagen(frame)
        
        # Predicciones
        pred_modelo1, conf_modelo1 = predecir_imagen(img_array, modelo1)
        pred_modelo2, conf_modelo2 = predecir_imagen(img_array, modelo2)
        
        # Convertir imagen a base64
        _, img_codificada = cv2.imencode('.jpg', frame)
        img_bytes = img_codificada.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return jsonify({
            'imagen': img_base64,
            'modelo1': {
                'prediccion': pred_modelo1,
                'confianza': round(conf_modelo1 * 100, 2),
                'resultado': pred_modelo1  
            },
            'modelo2': {
                'prediccion': pred_modelo2,
                'confianza': round(conf_modelo2 * 100, 2),
                'resultado': pred_modelo2  
            }
        })
    except Exception as e:
        print(f"Error en captura: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)