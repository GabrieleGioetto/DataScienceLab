import json

with open("to-bike.json") as f:
    data = json.load(f)

    activeStationCounter = 0
    freeBikesCounter = 0
    freeDocksCounter = 0
    for station in data["network"]["stations"]:
        if station["extra"]["status"] == "online":
            print(station)
            activeStationCounter += 1

        freeBikesCounter += station["free_bikes"]
        freeDocksCounter += station["empty_slots"]

    print(activeStationCounter, freeBikesCounter, freeDocksCounter)
