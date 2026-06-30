CREATE TABLE email_predictions (
    id SERIAL PRIMARY KEY,
    datum DATE,

    marke VARCHAR(100),
    modell VARCHAR(100),
    kraftstoff VARCHAR(50),
    getriebe VARCHAR(50),
    bundesland VARCHAR(100),

    verkaufszahl NUMERIC,
    hubraum_l NUMERIC,

    predicted_price NUMERIC,
    predicted_zufriedenheit INTEGER,
    predicted_category_2 INTEGER,
    predicted_category_3 INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);