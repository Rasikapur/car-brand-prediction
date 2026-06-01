import csv
import random

OUTPUT_FILE = "car_sales_data.csv"

BRANDS = [
    "Toyota", "Honda", "Ford", "BMW", "Mercedes",
    "Hyundai", "Nissan", "Volkswagen", "Audi", "Chevrolet",
    "Mazda", "Subaru", "Kia", "Tesla",
]
CAR_TYPES = ["Sedan", "SUV", "Hatchback", "Coupe", "Pickup", "Convertible"]
MONTHS = list(range(1, 13))


def choose_brand(price, car_type, month, year):
    if price >= 70000:
        return random.choices(["Mercedes", "BMW", "Audi", "Tesla"], [4, 4, 3, 3])[0]
    if price >= 45000:
        return random.choices(["BMW", "Audi", "Toyota", "Honda", "Subaru"], [3, 3, 3, 2, 2])[0]
    if car_type == "Pickup":
        return random.choices(["Ford", "Chevrolet", "Toyota"], [5, 4, 3])[0]
    if car_type == "SUV":
        return random.choices(["Toyota", "Honda", "Hyundai", "Ford", "Kia", "Nissan"], [4, 3, 3, 2, 2, 2])[0]
    if car_type == "Convertible":
        return random.choices(["Mazda", "BMW", "Ford"], [4, 3, 2])[0]
    if car_type == "Coupe":
        return random.choices(["BMW", "Audi", "Toyota", "Hyundai"], [4, 3, 3, 2])[0]
    if month in [3, 4, 9, 10]:
        return random.choices(["Toyota", "Honda", "Hyundai", "Kia", "Nissan"], [3, 3, 3, 2, 2])[0]
    if year >= 2022:
        return random.choices(["Hyundai", "Kia", "Toyota", "Honda", "Volkswagen"], [3, 3, 3, 2, 2])[0]
    return random.choices(["Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "Hyundai"], [4, 3, 3, 2, 2, 2])[0]


def generate_sales_data(start_year=2016, end_year=2025):
    header = ["year", "month", "price", "mileage", "car_type", "brand"]
    rows = []

    for year in range(start_year, end_year + 1):
        for month in MONTHS:
            for _ in range(80):
                price = random.choice([
                    15000, 18000, 20000, 22000, 25000, 28000,
                    30000, 32000, 35000, 38000, 40000, 42000,
                    45000, 48000, 50000, 55000, 60000, 65000,
                    70000, 75000, 80000, 90000, 100000, 120000,
                ])
                mileage = round(random.uniform(15.0, 50.0), 1)
                car_type = random.choices(
                    CAR_TYPES,
                    weights=[30, 35, 10, 8, 10, 7],
                    k=1,
                )[0]
                brand = choose_brand(price, car_type, month, year)
                rows.append([year, month, price, mileage, car_type, brand])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Generated {len(rows)} records at {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_sales_data()
