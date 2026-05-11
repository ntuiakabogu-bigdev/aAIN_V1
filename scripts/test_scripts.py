import asyncio, edge_tts

async def list_voices():
    voices = await edge_tts.list_voices()
    for v in voices:
        print(v["ShortName"], "-", v["Locale"], "-", v["Gender"])

asyncio.run(list_voices())
