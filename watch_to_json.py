import json
print("This will convert the ppg data from the watch into something a bit better looking")
input_filename = input('Please input feed name: ')
output_filename = input('Please input output feed name: ')
f = open(input_filename, 'r')
input_json = json.load(f)
clean_json = [{"value": point["value"], "time": point["time"]["$date"]} for point in input_json]
fout = open(output_filename, 'w+')
json.dump(clean_json, fout)
