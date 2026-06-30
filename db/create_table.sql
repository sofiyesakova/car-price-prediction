CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    datum DATE,
    marke VARCHAR(50),
    modell VARCHAR(100),
    preis_euro FLOAT,
    verkaufszahl INT,
    kraftstoff VARCHAR(30),
    getriebe VARCHAR(30),
    hubraum_l FLOAT,
    bundesland VARCHAR(50),
    kundenzufriedenheit FLOAT,
    jahr INT,
    monat INT,
    wochentag VARCHAR(20)
);