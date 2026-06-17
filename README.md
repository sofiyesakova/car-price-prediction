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


## Datenschutz & Datenquelle

Für dieses Projekt wird ein Datensatz verwendet, der auf öffentlich verfügbaren Statista-Daten zu Pkw-Neuzulassungen in Deutschland basiert. Die ursprünglichen Statistiken stammen vom Kraftfahrt-Bundesamt (KBA).

Der Datensatz enthält keine personenbezogenen Daten wie Namen, Telefonnummern, E-Mail-Adressen oder Adressen. Es werden ausschließlich aggregierte Fahrzeug- und Marktdaten verwendet.

Daher ist die Nutzung des Datensatzes DSGVO-konform und für Analyse- und Lernzwecke im Rahmen des Projekts geeignet.
