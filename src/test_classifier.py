from classifier import classify_news


test_articles = [

    (
        "NORMAL NEWS",
        """
        The Ministry of Health announced a new public healthcare
        program on Monday. The program will provide additional
        medical services to rural communities.
        """
    ),

    (
        "SENSATIONAL NEWS",
        """
        BREAKING!!! SHOCKING NEWS!!! You won't believe this!!!
        Scientists reveal an unbelievable secret that will change
        everyone's life!!!
        """
    ),

    (
        "UNIVERSITY NEWS",
        """
        The university announced that the annual examination
        schedule will be released next week. Students can check
        the official notice for further information.
        """
    )
]


print("\n========== NLP FAKE NEWS TESTING ==========\n")


for title, article in test_articles:

    score, classification, reasons = classify_news(article)

    print("------------------------------------------")
    print(f"Article Type       : {title}")
    print(f"Suspicion Score    : {score}")
    print(f"Classification     : {classification}")

    print("Reasons:")
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("  - No major suspicious linguistic patterns")

print("------------------------------------------")
print("\n==========================================")
