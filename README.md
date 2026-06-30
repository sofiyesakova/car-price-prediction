# Car Sales – Machine Learning für den deutschen Fahrzeugmarkt

## 1. Projektbeschreibung

Dieses Projekt kombiniert **Machine Learning** und **Large Language Models (LLMs)**, um Fahrzeugdaten automatisch zu analysieren und verschiedene Vorhersagen für den deutschen Automobilmarkt zu treffen.

Auf Basis von Fahrzeugmerkmalen können folgende Vorhersagen erstellt werden:

* **Fahrzeugpreis** (Regression)
* **Kundenzufriedenheit** bzw. ob der Kunde mit dem Fahrzeug voraussichtlich zufrieden sein wird (Klassifikation)
* **Marktsegment**, dem das Fahrzeug zugeordnet werden kann (Klassifikation)

Zusätzlich enthält das Projekt eine automatisierte E-Mail-Pipeline. Eingehende Fahrzeuganfragen werden mithilfe eines LLM analysiert, strukturierte Fahrzeugdaten extrahiert und anschließend an mehrere Machine-Learning-Modelle übergeben.

---

## 2. Datensatz

Der verwendete Datensatz enthält **Verkaufsdaten neuer bzw. nahezu neuer deutscher Fahrzeuge**, die **in Deutschland in den Jahren 2024 und 2025** angeboten oder verkauft wurden.

Die Daten umfassen unter anderem:

* Fahrzeugmarke und Modell
* Kraftstoffart
* Getriebe
* Bundesland
* Datum
* Verkaufszahl
* Hubraum
* Kundenzufriedenheit

---

## 3. Installation

Eine vollständige Installationsanleitung befindet sich in der Datei **INSTALL.md**.

---

## 4. Automatisierte E-Mail-Verarbeitung

Die Pipeline automatisiert die Verarbeitung eingehender E-Mails und verbindet ein Large Language Model mit mehreren Machine-Learning-Modellen.

### Workflow

1. Abruf ungelesener E-Mails über Gmail (IMAP)
2. Extraktion strukturierter Fahrzeuginformationen mit einem LLM
3. Umwandlung der Informationen in ein tabellarisches Format
4. Übergabe der Daten an die Machine-Learning-Modelle
5. Vorhersage von:

   * Fahrzeugpreis
   * Kundenzufriedenheit
   * Marktsegment
6. Speicherung der Ergebnisse in einer PostgreSQL-Datenbank

---

## 5. Verwendete Technologien

* Python
* Scikit-Learn
* Transformers (Hugging Face)
* Llama 3.1
* Qwen / Qwen2.5-3B-Instruct
* TinyLlama-1.1B-Chat-v1.0
* PostgreSQL
* SQLAlchemy
* Gmail IMAP
* Pandas
* NumPy

---

## 6. Projektstruktur

* `email_pipeline/` – Verarbeitung eingehender E-Mails
* `multi_pipeline/` – Multi-LLM-Pipeline. Ein Large Language Model analysiert eingehende E-Mails und übergibt die extrahierten Fahrzeugdaten an **vier trainierte Machine-Learning-Modelle**, die verschiedene Vorhersagen erzeugen.
* `models/` – trainierte Machine-Learning-Modelle
* `db/` – Datenbankstruktur und Datenbankanbindung
* `data/` – Enthält die Rohdaten sowie die bereinigten Daten im CSV-Format
* `notebooks/` – Datenanalyse, statistische Auswertung sowie Training und Evaluation der Machine-Learning-Modelle


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

=======

# 📊 Modellbewertung und Datenanalyse

## 📌 Überblick

Alle Machine-Learning-Modelle wurden auf demselben Feature-Set trainiert und miteinander verglichen.
Ziel war die Vorhersage des Fahrzeugpreises sowie der Kundenzufriedenheit auf Basis strukturierter Fahrzeugdaten.

## 📂 Datenbasis

Der verwendete Datensatz basiert auf öffentlich verfügbaren Statista-Daten zum deutschen Pkw-Markt, die ursprünglich auf Daten des Kraftfahrt-Bundesamts (KBA) beruhen.

👉 Quelle: https://de.statista.com/statistik/daten

📌 Wichtige Eigenschaften des Datensatzes:
Enthält ausschließlich neue oder nahezu neue Fahrzeuge
Fokus auf Markt- und Verkaufsdaten (2024–2025, Deutschland)
Keine Informationen zu:
   Baujahr
   Kilometerstand

👉 Dadurch basiert die Modellierung auf Marktmerkmalen statt Fahrzeuglebenszyklus-Daten.

## 📊 Modell-Setup

Für die Vorhersagen wurden folgende Modelle verwendet:

   Lineare Regression (Baseline)
   Random Forest Regressor
   Gradient Boosting Regressor
   Decision Tree Classifier (Kundenzufriedenheit)

## 📈 Ergebnisse

Alle Modelle wurden mit denselben Features trainiert (z. B. Marke, Modell und weitere verfügbare Marktmerkmale).

Modell	            MAE	       RMSE	      R²
Lineare Regression	13452.22	   16423.23 	0.6003
Gradient Boosting	   13452.24	   16423.20	   0.6003
Random Forest	      13459.48	   16416.15	   0.6007

👉 Bestes Modell: Random Forest Regressor    

      R² ≈ 0.60

## 📂 Analyse-Dateien

Die detaillierte Analyse und Visualisierungen befinden sich in folgenden Dateien:

notebooks/Data_clearning.ipynb
notebooks/descriptive_statistics.ipynb
notebooks/Inferenzstatistik.ipynb
notebooks/Einflussanalyse_Preis_Euro.ipynb
notebooks/Einflussanalyse_Kundzufrid.ipynb
notebooks/Einflussanalyse_Preissegmenten.ipynb


## 📌 Interpretation (Kurzfassung)

   Alle Modelle erreichen ähnliche Ergebnisse (R² ≈ 0.60)
   Das Fahrzeugmodell ist der stärkste Einflussfaktor
   Zusätzliche Features haben nur geringe Vorhersagekraft
   Der limitierende Faktor ist die Datenbasis, nicht das Modell

## 🚀 Fazit

   Solide Modellleistung für Marktbasierte Daten
   Gute Vorhersage auf aggregierter Ebene möglich
   Keine signifikanten Unterschiede zwischen Modellen

## 🔮 Zukünftige Verbesserungen

Für zukünftige Versionen sind folgende Erweiterungen geplant:

Integration zusätzlicher Fahrzeugdaten:
      Kilometerstand
      Baujahr
      Motorleistung (PS)
      Zustand / Ausstattung
      Erweiterung der Datengrundlage über mehrere Jahre
      Entwicklung einer Preisrange-Vorhersage:
         minimal erwartbarer Preis
         maximal erwartbarer Preis

 👉 Ziel: realistischere Marktpreis-Spanne statt einzelner Punktprognose
