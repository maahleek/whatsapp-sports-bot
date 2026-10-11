from sportybet import extract_booking, normalize_booking_code


def test_normalize_booking_code():
    assert normalize_booking_code(" hw8em8 ") == "HW8EM8"


def test_extract_booking_drops_account_identity_and_reads_selection():
    payload = {
        "bizCode": 10000,
        "message": "Success",
        "data": {
            "shareCode": "HW8EM8",
            "shareURL": "http://www.sportybet.com/ng/?shareCode=HW8EM8",
            "userId": "should-not-be-exposed",
            "deadline": 1787666400000,
            "betType": "SINGLE",
            "unavailableOutcomes": [],
            "outcomes": [
                {
                    "eventId": "sr:match:67015328",
                    "estimateStartTime": 1786802400000,
                    "matchStatus": "Not start",
                    "homeTeamName": "KFUM Oslo",
                    "awayTeamName": "Lillestroem SK",
                    "sport": {
                        "category": {
                            "name": "Norway",
                            "tournament": {"name": "Eliteserien"},
                        }
                    },
                    "markets": [
                        {
                            "id": "1",
                            "name": "1X2",
                            "outcomes": [
                                {
                                    "id": "1",
                                    "odds": "2.96",
                                    "isActive": 1,
                                    "desc": "Home",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }

    booking = extract_booking(payload)
    assert booking["share_code"] == "HW8EM8"
    assert "userId" not in booking
    assert len(booking["selections"]) == 1

    selection = booking["selections"][0]
    assert selection["home_team"] == "KFUM Oslo"
    assert selection["away_team"] == "Lillestroem SK"
    assert selection["competition"] == "Eliteserien"
    assert selection["market_name"] == "1X2"
    assert selection["outcome_name"] == "Home"
    assert selection["odds"] == 2.96


def main():
    test_normalize_booking_code()
    test_extract_booking_drops_account_identity_and_reads_selection()
    print("SportyBet parser tests passed.")


if __name__ == "__main__":
    main()
