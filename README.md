# Car Sales

## 1. Project description

## 2. Dataset Info

## 1. 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/sofiyesakova/car-price-prediction
cd car-price-prediction
```

2. Create virtual environment:

```bash
python -m venv venv
```

3. Activate virtual environment:

   Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. # Automatisierte E-Mail-Auswertung für Fahrzeugverkaufsdaten

## Projektbeschreibung

Dieses Projekt kombiniert Machine Learning und Large Language Models (LLMs), um eingehende E-Mails automatisch zu analysieren und Fahrzeugpreisdaten vorherzusagen.

Der Workflow besteht aus mehreren Schritten:

1. Abruf ungelesener E-Mails über Gmail (IMAP)
2. Extraktion strukturierter Informationen mittels Llama 3.1
3. Umwandlung der Daten in ein tabellarisches Format
4. Vorhersage des Fahrzeugpreises mit einem trainierten Machine-Learning-Modell
5. Speicherung der Ergebnisse in einer PostgreSQL-Datenbank

## Verwendete Technologien

* Python
* Scikit-Learn
* Transformers (Hugging Face)
* Llama 3.1
* PostgreSQL
* SQLAlchemy
* Gmail IMAP
* Pandas und NumPy

## Projektstruktur

* `email_pipeline/` – automatisierte E-Mail-Verarbeitung
* `models/` – trainierte Machine-Learning-Modelle
* `DB/` – Datenbankanbindung
* `notebooks/` – Training und Experimente

## Beispiel einer E-Mail

Datum: 2026-06-16

Marke: BMW

Modell: X5

Kraftstoff: Diesel

Getriebe: Automatik

Hubraum_l: 3.0

Verkaufszahl: 4

Kundenzufriedenheit: 5

## Ausführung

```bash
python -m email_pipeline.main
```
=======
# 🌐 Web Application

Die Webanwendung wurde mit Flask, HTML, CSS und JavaScript entwickelt.  
Sie ermöglicht Nutzern die Eingabe von Fahrzeugdaten und erstellt mithilfe trainierter Machine-Learning-Modelle sowohl eine Fahrzeugpreisprognose als auch eine Vorhersage der Kundenzufriedenheit.

Zusätzlich verfügt die Anwendung über ein mehrstufiges Eingabeformular, eine Kontaktseite mit KI-Analyse sowie einen integrierten Chatbot, der Fahrzeuginformationen aus Nutzereingaben erkennt und passende Prognosen ausgibt.

## Run the Application

Wechseln Sie in das Stammverzeichnis des Projekts und führen Sie folgenden Befehl aus:

```bash
python app1.py
```

Nach dem Start der Anwendung kann die Webanwendung im Browser unter folgender Adresse geöffnet werden:

http://localhost:5005

## Verwendung der Anwendung

Die Webanwendung bietet mehrere Funktionen:

1. Fahrzeugpreis vorhersagen
2. Kundenzufriedenheit vorhersagen
3. KI-gestützte Fahrzeuganalyse über das Kontaktformular
4. Mehrstufiges Eingabeformular zur Prognose
5. Integrierter Chatbot zur Analyse von Fahrzeuginformationen

Benutzer können Fahrzeugdaten eingeben und erhalten innerhalb weniger Sekunden eine Preis- und Zufriedenheitsprognose.

## Verwendete Machine-Learning-Modelle

- Random Forest Regressor für die Fahrzeugpreisprognose
- Decision Tree Classifier für die    Kundenzufriedenheitsprognose

## Datenschutz & Datenquelle

Für dieses Projekt wird ein Datensatz verwendet, der auf öffentlich verfügbaren Statista-Daten zu Pkw-Neuzulassungen in Deutschland basiert. Die ursprünglichen Statistiken stammen vom Kraftfahrt-Bundesamt (KBA).

Der Datensatz enthält keine personenbezogenen Daten wie Namen, Telefonnummern, E-Mail-Adressen oder Adressen. Es werden ausschließlich aggregierte Fahrzeug- und Marktdaten verwendet.

Daher ist die Nutzung des Datensatzes DSGVO-konform und für Analyse- und Lernzwecke im Rahmen des Projekts geeignet.

# 📊 Modellbewertung der Ergebnisse

## 🔍 Vergleich der Modelle

Alle Modelle wurden mit demselben Feature-Set trainiert:

- `marke`
- `modell`

Folgende Modelle wurden evaluiert:

- Lineare Regression
- Random Forest
- Gradient Boosting

### Ergebnisse:

| Modell             |      MAE |     RMSE |     R² |
| ------------------ | -------: | -------: | -----: |
| Lineare Regression | 13452.22 | 16423.23 | 0.6003 |
| Gradient Boosting  | 13452.24 | 16423.20 | 0.6003 |
| Random Forest      | 13459.48 | 16416.15 | 0.6007 |

---

## 🏆 Bestes Modell

Das beste Modell ist:

**Random Forest Regressor**

- **MAE:** 13.459,48
- **RMSE:** 16.416,15
- **R²:** 0.6007

Die Unterschiede zwischen den Modellen sind jedoch sehr gering.

---

# 📈 Interpretation der Ergebnisse

## ⚠️ Zentrale Beobachtung

Alle getesteten Modelle zeigen nahezu identische Ergebnisse:

- R² ≈ **0.60**
- MAE ≈ **13.400–13.500**
- RMSE ≈ **16.400**

Das zeigt eindeutig, dass die Modellwahl nur einen sehr geringen Einfluss auf die Performance hat.

---

## 🧠 Warum ist die Leistung begrenzt?

### 1. Starke Dominanz des Features `modell`

Das Fahrzeugmodell erklärt bereits einen großen Teil der Preisvariation.

Jedes Modell hat einen relativ stabilen durchschnittlichen Preisbereich, zum Beispiel:

- `Corsa` → ~29.000 €
- `Passat` → ~42.000 €
- `E-Klasse` → ~82.000 €

👉 Das bedeutet, dass das Modell bereits den größten Teil des Preissignals enthält.

---

### 2. Geringer Einfluss der anderen Features

Andere verfügbare Merkmale (z. B. Kraftstoffart, Getriebe, Verkaufszahl, Kundenzufriedenheit usw.) zeigen **sehr geringe oder nahezu keine Korrelation mit dem Zielwert (Preis)**.

In der Praxis bedeutet das:

> Diese Features tragen nur minimal zur Erklärung der Preisunterschiede bei.

Daher verbessert ihre Nutzung die Modellleistung kaum.

---

### 3. Hohe Streuung innerhalb desselben Modells

Selbst innerhalb eines Fahrzeugmodells gibt es große Preisunterschiede:

- C-Klasse: ±18.000 €
- E-Klasse: ±22.000 €

Diese Variabilität kann mit den aktuellen Features nicht vollständig erklärt werden.

---

### 4. Das Modell ist nicht der limitierende Faktor

Da alle Modelle nahezu gleich gut performen, zeigt sich:

> Die Begrenzung liegt in den Daten, nicht in den Algorithmen.

Komplexere Modelle können die Leistung nicht verbessern, wenn keine zusätzlichen relevanten Features vorhanden sind.

---

# 📌 Fazit

- Alle Modelle erreichen ein Leistungsplateau bei ca. R² = 0.60
- Das Feature `modell` ist der wichtigste Prädiktor
- Weitere Features haben nur geringe Vorhersagekraft
- Die Modellwahl hat kaum Einfluss auf die Endleistung

---

# 🚀 Zukünftige Verbesserungen

Um die Vorhersagequalität zu verbessern, sollten zusätzliche relevante Merkmale integriert werden:

- Kilometerstand
- Motorleistung (PS)
- Fahrzeugzustand
- Anzahl der Vorbesitzer
- Ausstattung / Trim-Level
- Unfallhistorie

Diese Merkmale würden die Modellgenauigkeit voraussichtlich deutlich erhöhen.
