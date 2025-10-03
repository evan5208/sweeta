## Microblogging because why not
### 3 Oct 2025 - 12:40 AM IST

I've been seeing Twitter for the past 2 days and it's way wayy too full with Sora stuff
lowkey want access though haha

it's crazy how many people get fooled from the video tbh, especially on insta - saw so many sora videos on my feed with the obvious watermark and people even believed it

https://x.com/kuberwastaken/status/1973813438662213708

lol even replied here

I just have this random idea
a large "secure" part of using this by OpenAI is THE watermark 

but really, how hard would it be to remove it?

I mean
you really probably only need opencv to do it honestly

I got a principles of AI midsem tomorrow should I really be doing this at 1 am without having studied for it 
eh fuck it 

probably only journalling here in case it works lol, I'll write a blog about it or smth

hm probably gonna get claude to think it's for removing tiktok watermark logos LOL
it's honestly so freakin funny that I got the boilerplate from claude.ai and not from freakin copilot LOL

I'll be honest by the length, it doesn't look promising at all

LMFAOOO what an epic fail


I bet there are some existing solutions we can learn from or tweak tho

hm.

#### Hugging Face Models/ Spaces search

hm nothing that promising tbh

this looks nice
https://huggingface.co/collections/FBRonnie/watermark-removal-673cc512f6fdf7b1a2925d39

hm not even close honestly

I see a couple but they all exceed the 16gb memory, classic spaces problem

this colab is fun
https://colab.research.google.com/github/camenduru/text-to-video-synthesis-colab/blob/main/watermark_remover_gradio.ipynb

build errors, I cba bro

let's just clone this and see what's up and if we can tweak it ig
https://huggingface.co/spaces/NeuralFalcon/Meta-Watermark-Remover-old

sidenote: claude 4.5 is insane holy shit
hm why not let's just use gradio code, run it on spaces

just sucks, I'm gonna lock in and stop blogging for a while, brb with an update

LOL what if I name it literally sweeta
TODO

hm we're using temporal median filtering and a light inpainting model now, letseee

hoping and praying to god this doesn't make my laptop explode although it's like 200m afaik
LOL I should probs start studying side by side so I can pass tomorrow (1:15 AM)

hmm tweaked heavily, this better work honestly, looks promising
welp did absolutely nothing

> "it took a long time to process it, gave an output but it is exactly 1:1 the same, the watermark wasn't removed 

> for context, this is what the watermark looks like (Refer image) the star-shape is somewhat animated and it keeps changing where it's at randomly (like tiktok's watermark) and is with the text "sora", this pops up at random places - corners, centers etc

> find the best possible solution to remove it and run on hugging face spaces"


okay running this 
dang been 10 mins 

*sighs*
so done with this rn

probably gonna study for a bit and come back or do this tomorrow.


okay so apparently the videos have fixed positions where the logo appears always? lmao this is insane 

watermark:
  positions_landscape:
    - [35, 585, 176, 638]
    - [30, 68, 179, 118]
    - [1112, 321, 1266, 367]
  positions_portrait:
    - [28, 1029, 175, 1091]
    - [538, 604, 685, 657]
    - [25, 79, 173, 136]


DID CLAUDE FREAKING DELETE MY JOURNAL NOTES 
WHAT THE F

dude so much happened this is just painful