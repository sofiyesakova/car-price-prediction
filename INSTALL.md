# Installationsanleitung
## 1. Projekt herunterladen
Repository klonen oder Projekt herunterladen:
```bash
git clone <https://github.com/sofiyesakova/car-price-prediction.git>
```
Anschließend in das Projektverzeichnis wechseln:
```bash
cd <car-price-prediction>
```
---
## 2. Virtuelle Python-Umgebung erstellen
Es wird empfohlen, eine virtuelle Umgebung zu verwenden.

```bash
python -m venv .venv
```
---
## 3. Virtuelle Umgebung aktivieren

**Windows**
```bash
.venv\Scripts\activate
```
**Linux / macOS**
```bash
source .venv/bin/activate
```
---
## 4.  Python-Abhängigkeiten installieren
Alle für das Projekt erforderlichen Bibliotheken werden über die Datei `requirements.txt` installiert.
```bash
pip install -r requirements.txt
```
## 5. PostgreSQL installieren
- pgAdmin 4 offnen

## 6. .env and config.py erstellen
- .env.example
- config.py.example

## 7. Datenbank importieren
- db/create_table.sql
- multi_pipeline/create_email_predictions.sql
```bash
 python -m db.data_loader  
```

## 8. ML-Modelle trainieren

Führen Sie die Trainingsskripte aus, um die benötigten ML-Modelle zu erstellen.

```bash
python notebooks/random_forest.py
python notebooks/klassifikation.py
python notebooks/train_rf_fahrzeugpreiskategorie_2klassen.py
python notebooks/train_rf_fahrzeugpreiskategorie_3klassen.py
```

## 9. LLM Modelle herunterladen
- Qwen/Qwen2.5-3B-Instruct oder TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Beispiel einer E-Mail
Sehr geehrte Damen und Herren, 
ich bin aktuell auf der Suche nach einer Mercedes-Benz C-Klasse mit einem 4.0-Liter-Motor (V8 / AMG-Modell) und Automatikgetriebe. Da Sie ein passendes Fahrzeug in Hessen anbieten, wende ich mich direkt an Sie.

Besonders wichtig sind mir bei meiner Suche zwei spezifische Ausstattungswünsche: Das Fahrzeug sollte eine rote Außenlackierung sowie eine hochwertige Lederausstattung besitzen.

Oder:

Marke: BMW
Modell: X5
Kraftstoff: Diesel
Getriebe: Automatik
Hubraum_l: 3.0
Verkaufszahl: 1

## Ausführung
```bash
python -m multi_pipeline.pipeline
```


# 10. Flask starten

# 11. Streamlit starten