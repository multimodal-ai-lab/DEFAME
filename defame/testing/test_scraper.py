from defame.evidence_retrieval import scraper


def test_firecrawl_is_running():
    assert scraper.firecrawl_url is not None


def test_scraper():
    urls_to_scrape = [
        "https://www.washingtonpost.com/video/national/cruz-calls-trump-clinton-two-new-york-liberals/2016/04/07/da3b78a8-fcdf-11e5-813a-90ab563f0dde_video.html",
        "https://nypost.com/2024/10/11/us-news/meteorologists-hit-with-death-threats-after-debunking-hurricane-conspiracy-theories/",
        "https://www.tagesschau.de/ausland/asien/libanon-israel-blauhelme-nahost-102.html",
        "https://www.zeit.de/politik/ausland/2024-10/wolodymyr-selenskyj-berlin-olaf-scholz-militaerhilfe",
        "https://edition.cnn.com/2024/10/07/business/property-damange-hurricane-helene-47-billion/index.html"
        "https://www.getrichslowly.org/wp-content/uploads/5693979929_1cdcdcc674.jpg",
    ]
    for url in urls_to_scrape:
        result = scraper.scrape(url)
        assert result is not None
        print(result, "\n\n\n")


def test_excluded_urls():
    scraper.allow_fact_checking_sites = False
    urls_to_scrape = [
        "https://www.politifact.com/factchecks/2025/mar/05/donald-trump/fact-checking-trumps-claim-that-a-group-headed-up/",
        "https://www.reddit.com/r/Jordan_Peterson_Memes/comments/1j4eump/i_hope_they_get_well_soon/",
        "https://travel.state.gov/content/travel.html",
    ]
    for url in urls_to_scrape:
        result = scraper.scrape(url)
        assert result is None
