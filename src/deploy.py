from pipeline import house_price_pipeline

if __name__ == "__main__":
    house_price_pipeline.serve(
        name="house-price-30sec-schedule",
        interval=30,
    )