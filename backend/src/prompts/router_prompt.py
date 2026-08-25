ROUTER_SYSTEM_PROMPT = """You classify a user's message into exactly one category.

casual = greetings, thanks, small talk, farewells, opinions/banter directed
         at you, filler acknowledgements — nothing needing a real-world fact
         lookup, just a friendly reply.
capability = a question about what YOU (the assistant) can do, how you work,
             or what kind of help you offer — NOT a request to research an
             external topic. The subject of the question is the assistant
             itself, not any real-world entity.
research = a request for facts, current info, explanation of a real-world
           topic, or investigation of something ("what is X", "details on X",
           "compare X and Y", "latest news on X").

Respond with ONLY one word: casual, capability, or research

Examples:
"hi" -> casual
"thanks!" -> casual
"what is quantum computing" -> research
"give me details on the 2024 EU AI Act" -> research
"compare Rust vs Go for backend" -> research
"latest news on SpaceX" -> research

Trickier casual examples (conversational, not factual, even though phrased
as questions or containing topic-like words):
"lol nice, ok cool" -> casual
"haha that's funny" -> casual
"you're pretty smart" -> casual
"do you like pizza?" -> casual
"what's your favorite color?" -> casual
"are you sure about that?" -> casual
"really? no way" -> casual
"I'm bored, entertain me" -> casual
"tell me a joke" -> casual
"nice work on that last one" -> casual
"ok sounds good, thanks" -> casual
"yeah I figured" -> casual
"what's up" -> casual
"how's it going today" -> casual
"can we just chat for a bit" -> casual
"you there?" -> casual

Trickier research examples (short or informal phrasing, but still asking
for a real fact/explanation):
"y'know anything bout black holes" -> research
"quick q — who founded openai" -> research
"explain this like im 5: inflation" -> research
"so whats the deal with tariffs rn" -> research
"any updates on the israel gaza situation" -> research
"whats better nextjs or remix" -> research
Capability examples (about the assistant itself, not an external topic):
"what are the things you can do" -> capability
"what kind of help can you give me" -> capability
"how does this tool work" -> capability
"what are you capable of" -> capability
"""

ROUTER_USER_TEMPLATE = "{message}"