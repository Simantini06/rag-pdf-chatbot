"""Generate data/sample_weather_station_guide.pdf - a short, entirely fictional product manual.

The content is original and invented for demo purposes only. Requires reportlab
(pip install reportlab); the generated PDF is committed so you normally don't need to run this.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_weather_station_guide.pdf"

PAGES = [
    (
        "Orbit-7 Home Weather Station - User Guide",
        [
            "This guide describes the Orbit-7, a fictional home weather station invented for this demo. "
            "The Orbit-7 has two parts: an outdoor sensor mast and an indoor display console. The sensor mast "
            "measures temperature, relative humidity, wind speed, wind direction and rainfall. The console shows "
            "live readings and keeps a 30-day history.",
            "Box contents: one sensor mast, one display console, a mounting bracket with four screws, a USB-C "
            "power cable for the console and this guide. The sensor mast runs on two AA lithium batteries, which "
            "are not included.",
            "Placement. Mount the sensor mast at least 1.5 metres above the ground and at least 3 metres away "
            "from walls, trees and air-conditioner exhausts. A rooftop or an open garden pole works well. Avoid "
            "placing the mast directly above concrete or a metal roof, because reflected heat makes temperature "
            "readings too high on sunny afternoons.",
        ],
    ),
    (
        "Setup and Pairing",
        [
            "Step 1: insert the two AA lithium batteries into the sensor mast. The status LED blinks green three "
            "times to confirm power. Step 2: connect the console to power with the USB-C cable. Step 3: hold the "
            "PAIR button on the back of the console for five seconds until the radio icon flashes.",
            "The console then searches for the sensor mast for up to three minutes. Keep the two units within "
            "10 metres of each other during pairing. Once paired, the radio icon stops flashing and readings "
            "appear within one minute. The wireless link uses the 868 MHz band and has a range of about 100 "
            "metres in open space, or roughly 30 metres through two brick walls.",
            "Battery life. With lithium cells the sensor mast runs for about 18 months. Alkaline cells are "
            "supported but last roughly 8 months and perform poorly below minus 10 degrees Celsius. When the "
            "battery is low, a battery symbol appears next to the outdoor temperature on the console.",
        ],
    ),
    (
        "Troubleshooting, Maintenance and Warranty",
        [
            "No outdoor readings: first check the sensor batteries, then move the console closer to the mast and "
            "repeat pairing. If readings still do not appear, perform a factory reset by holding the PAIR and "
            "MODE buttons together for ten seconds. A factory reset clears the 30-day history.",
            "Rain gauge reading too low: the funnel is probably blocked by leaves or dust. Clean the funnel and "
            "the tipping bucket with water every three months. Never use oil or detergent on the tipping bucket, "
            "because residue changes how much water it takes to tip.",
            "Wind speed stuck at zero: check that the cups spin freely. Spider webs and bird droppings are the "
            "most common causes. The anemometer cups can be replaced; the spare part code is ORB7-CUP.",
            "Warranty. The Orbit-7 comes with a two-year limited warranty covering manufacturing defects. "
            "The warranty does not cover damage from lightning, physical impact or opening the sealed sensor "
            "housing. To make a claim, keep your proof of purchase and quote the serial number printed under "
            "the battery cover.",
        ],
    ),
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Title"] if i == 0 else styles["Heading1"]))
        story.append(Spacer(1, 12))
        for p in paras:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=A4, title="Orbit-7 User Guide (fictional sample)", author="Simantini Das").build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
