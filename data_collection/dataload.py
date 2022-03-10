import requests
import time
import os
import sys
from datetime import datetime
import json
import csv

def readData(rawData):
    #["routeID", "timestamp", "stopName", "routeShortName", "serviceDay", "scheduledArrival", "arrivalDelay"]
    #stoptimes = rawData.get("data").get("stops")[0].get("stoptimesWithoutPatterns")
    stops = rawData.get("data").get("stops")
    dataLoad = []
    for stop in stops:
        stoptimes = stop.get("stoptimesWithoutPatterns")
        for arrivals in stoptimes:
            if arrivals.get("realtime") is True:
                #check index of the stop
                indexOfStop = -1
                trip = arrivals.get("trip")
                tripStops = trip.get("stops")
                try:
                    indexOfStop = tripStops.index({"code": stop.get("code")})
                except:
                    indexOfStop = -1
                dataLoad.append([trip.get("id"), time.time(), stop.get("code"), trip.get("routeShortName"), arrivals.get("serviceDay"), arrivals.get("scheduledArrival"), arrivals.get("arrivalDelay"), indexOfStop, len(tripStops)])

    return dataLoad


def processData(data, waitList):
    publish = []
    for line in waitList:
        if not any(line[0] in sublist[0] for sublist in data):
            publish.append(line)
            
    return publish


def publishData(writer, lines):
    for line in lines:
        print(line)
        writer.writerow(line)


def monitor(pathName, stopNames):
    url = 'https://api.digitransit.fi/routing/v1/routers/hsl/index/graphql'
    headobj = { "Content-Type": "application/json" }
    query = """query 
        {
            stops(name: "%s") {
                name
                code
                stoptimesWithoutPatterns(numberOfDepartures: 15) {
                    scheduledArrival
                    arrivalDelay
                    realtime
                    serviceDay
                    trip {
      			        id
                        routeShortName
                        stops {
                            code
                        }
      			    }
                }
            }
        }""" % ' '.join(stopNames)

    waitList = []
    while True:
        x = requests.post(url, json={'query': query}, headers = headobj)
        if x.status_code == 200:
            f = open(pathName , 'a')
            writer = csv.writer(f)
            data = readData(x.json())
            publish = processData(data, waitList)
            waitList = data
            publishData(writer, publish)
            f.close()
        else:
            print("Error. Try again.")

        time.sleep(15)


def createMonitor(stopNames, stopSet):
    fileName = "data_" + stopSet + "_" + str(time.time()) + ".csv"
    pathName = "data/data/" + fileName
    f = open(pathName , 'w')
    writer = csv.writer(f)
    header = ["routeID", "timestamp", "stopName", "routeShortName", "serviceDay", "scheduledArrival", "arrivalDelay","distanceFromStart","stopsTotal"]
    writer.writerow(header)
    f.close()
    monitor(pathName,stopNames)


def main():
    stopNames = ["H3005", "H2569", "H3008", "H3245", "H3120", "H2205", "H4298", "H2165", "H2126"]
    #stopNames = ["H1911", "H1914", "H1379", "E2332", "H1234", "H1025", "H1222", "H1210", "H1231"]
    stopSet = "set1"
    #stopSet = "set2"
    createMonitor(stopNames, stopSet)
    return None


if __name__ == "__main__":
    main()