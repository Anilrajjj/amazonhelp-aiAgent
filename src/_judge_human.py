"""Human ratings for the 75 replies in results/judge_sample.jsonl.

One boolean per sampled reply, in file order: would I let this go out publicly
with no human reading it first?

The sample is blind -- it mixes replies from all three systems, shuffled, with
the producing system not shown while rating. It was rated before the judge was
run on it.

The rule I held myself to, written down first and applied in order:

  FALSE if the reply fabricates anything, or asserts an action or a fact the
        brand cannot know from Twitter ("we've responded to you here", "the
        e-mail sent by our Account Specialist").
  FALSE if it invites account or payment details into a public thread.
  FALSE if it is a topic-blind generic template sent to a message where it is
        obviously wrong -- praise, a joke, a catalogue request, a pre-purchase
        question. These read as broken automation.
  FALSE if it is tone-deaf to what the customer just said: telling someone who
        said they cannot reach support to contact support, or asking "has the
        delivery date been missed?" of someone who said it was a month ago.
  FALSE if it is an incoherent fragment (the "2/2" second halves of multi-part
        brand tweets that leak in through nearest-neighbour retrieval).
  TRUE  otherwise, including bland-but-safe generic replies sent to a genuine
        problem. That is what this brand actually sends, and holding a draft to
        a higher standard than the brand's own replies would make the number
        meaningless.

Note the deliberate asymmetry: a generic "sorry, contact us here" is TRUE on a
real complaint and FALSE on a joke. The failure being measured is
inappropriateness, not blandness.
"""
HUMAN_RATINGS = [
    True,    # 0  wrong item received -- generic but safe and on-topic
    True,    # 1  Pantry feature suggestion -- acknowledged and routed
    True,    # 2  charge on a closed account -- responsive, no fabrication
    False,   # 3  Kannada catalogue request -- "sorry for the trouble" is wrong
    True,    # 4  vague account request -- asks for detail without inviting PII
    False,   # 5  profane shipping rant -- "Thanks for reaching out", no apology
    False,   # 6  giveaway photo (positive) -- apology template is broken here
    False,   # 7  asked for an email address -- deflects the actual question
    True,    # 8  masks never arrived
    False,   # 9  Prime Video content request -- apology template is wrong
    True,    # 10 gift card unfulfilled
    False,   # 11 asks the customer to post the amount and card reference publicly
    False,   # 12 simple credit question -- "sorry for the trouble" is wrong
    False,   # 13 Twitch affiliate question -- apology template is wrong
    True,    # 14 unsafe parcel placement
    True,    # 15 same profane rant, but apologises and asks a useful question
    True,    # 16 fraud on saved card -- names the issue, routes to reporting
    False,   # 17 says "contact us" to someone who said they cannot reach support
    True,    # 18 catalogue request answered as a catalogue request
    True,    # 19 angry gifting complaint -- apologises, asks without inviting PII
    True,    # 20 duplicate item sent
    False,   # 21 "the link mentioned above" -- there is no link in the reply
    False,   # 22 praise for Alexa -- apology template is wrong
    False,   # 23 leaked "^MP(2/2)" sign-off and fragment
    True,    # 24 items not dispatched
    True,    # 25 packaging complaint routed to the packaging team
    True,    # 26 defective product -> return/replacement
    True,    # 27 got RAM instead of an SSD
    True,    # 28 asks the one question that determines the path (seller vs Amazon)
    False,   # 29 stock question -- "sorry for the trouble" is wrong
    False,   # 30 asks "have you checked the mailbox" of someone who said it was there
    True,    # 31 profane cancellation threat -- apologises, asks a useful question
    True,    # 32 gift card unfulfilled
    True,    # 33 refund process question
    True,    # 34 warranty claim
    False,   # 35 invents "the e-mail sent by our Account Specialist"
    False,   # 36 joke Alexa request -> communication preferences, irrelevant
    True,    # 37 cancelled gift -- empathetic and specific
    True,    # 38 affiliate question -- generic but offers a real channel
    False,   # 39 "We've responded to your query here" -- asserts an unknowable action
    False,   # 40 "contact us" to someone who said no one will talk to them
    True,    # 41 two-month refund delay -- apologises and routes
    False,   # 42 joke about a switched order -- apology template is wrong
    True,    # 43 refund timing explained without promising a date
    True,    # 44 defective product
    True,    # 45 app sync
    True,    # 46 preorder not dispatched
    True,    # 47 delivered-but-missing -> locate steps
    False,   # 48 GSTIN invoice question -- "sorry for the trouble" is wrong
    True,    # 49 Fire TV X-ray -- asks for the error
    False,   # 50 "contact us" after they said the chat rep was no help
    True,    # 51 refund complaint -- explicitly warns against posting account info
    True,    # 52 profane gift-card rant -- apologises, asks a scoping question
    True,    # 53 German puzzle -- answered correctly in German
    True,    # 54 earphones refund
    True,    # 55 package late -> identify carrier
    False,   # 56 incoherent "2/2" fragment as a standalone reply
    True,    # 57 Prime Video error -- asks for the message
    True,    # 58 damaged box, fourth time -- names it, offers return/replacement
    True,    # 59 gift card unfulfilled
    True,    # 60 Fire Stick update
    True,    # 61 vague account request handled safely
    True,    # 62 £150 parcel lost
    True,    # 63 refund outstanding since August
    True,    # 64 cancelled order, payment not returned
    True,    # 65 two failed deliveries -- names it
    True,    # 66 packaging feedback
    True,    # 67 three bad orders -- generic but on-topic
    False,   # 68 "I didn't understand that" to a perfectly clear message
    True,    # 69 praise answered as praise
    False,   # 70 asks if the date was missed; they said it was a month ago
    False,   # 71 promo spam -> leaked "2/2" fragment, irrelevant
    True,    # 72 conflicting courier updates -- asks what the agent said
    False,   # 73 pre-purchase Echo question -- apology template is wrong
    True,    # 74 catalogue request answered as a catalogue request
]
