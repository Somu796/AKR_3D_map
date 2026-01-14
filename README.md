<h1 align="center"> 📡 🌍 AKR 3D Map 📻 🌐 </h1>

**Work in Progress as of Dec 2025**

Here we provide code to read in various AKR event lists, and map their occurrence in the near-Earth space environment.

# Acknowledgements

* SH's work at DIAS was supported by a SCOSTEP PRESTO Database Construction grant entitled "AKR as a Barometer for Space Weather: a new, interactive map".
* [ARF](https://github.com/arfogg)'s work at DIAS was supported by Taighde Éireann - Research Ireland Laureate Consolidator award SOLMEX to [CMJ](https://github.com/caitrionajackman).

<p align="center">
<img src="assets/SCOSTEP_logo.png" width="200">
<img src="assets/PRESTO_logo.png" width="100">
<img src="assets/Research_Ireland_RGB_logo_green.webp" width="200">
</p>

# Thinking process

CSV File (Python conversion) ->  TFCat JSON (dates as ISO strings) -> (MongoDB insert with conversion) -> MongoDB BSON (dates as ISODate objects) (Application query) ->  Python datetime objects / JavaScript Date objects -> (Analysis/Display)

# Good Practices will be followed all over

It should be.

1. OOP
2. Type check
3. Write pytests
4. Try and Catch error handling.
