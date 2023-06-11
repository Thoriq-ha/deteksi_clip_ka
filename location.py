import geocoder


a = geocoder.ip('me').latlng
# a = [1, 2]
print(f"https://maps.google.com/?q={a[0]},{a[1]}")
