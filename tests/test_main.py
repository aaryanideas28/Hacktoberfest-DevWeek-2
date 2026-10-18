from main import extract_items

def test_decision_extraction():
    result = extract_items("We decided to use SQLite.")

    assert result["decisions"] == [
        "We decided to use SQLite"
    ]


def test_completed_work():
    result = extract_items("The endpoint is completed.")

    assert result["completed_work"] == [
        "The endpoint is completed"
    ]


def test_unresolved_issue():
    result = extract_items("The login endpoint is pending.")

    assert result["unresolved_issues"] == [
        "The login endpoint is pending"
    ]


def test_next_step():
    result = extract_items("Next, implement authentication.")

    assert result["next_steps"] == [
        "Next, implement authentication"
    ]


def test_multiple_categories():
    conversation = (
        "We decided to use SQLite. "
        "The endpoint is completed. "
        "The login endpoint is pending. "
        "Next, implement authentication."
    )

    result = extract_items(conversation)

    assert len(result["decisions"]) == 1
    assert len(result["completed_work"]) == 1
    assert len(result["unresolved_issues"]) == 1
    assert len(result["next_steps"]) == 1


def test_empty_conversation():
    result = extract_items("")

    assert result["decisions"] == []
    assert result["completed_work"] == []
    assert result["unresolved_issues"] == []
    assert result["next_steps"] == []


def test_unrecognized_text():
    result = extract_items("Hello, how are you?")

    assert result["decisions"] == []
    assert result["completed_work"] == []
    assert result["unresolved_issues"] == []
    assert result["next_steps"] == []


def test_not_completed_is_not_completed_work():
    result = extract_items("The task is not completed.")

    assert result["completed_work"] == []
    assert result["unresolved_issues"] == [
        "The task is not completed"
    ]


def test_not_finished_is_not_completed_work():
    result = extract_items("The feature is not finished.")

    assert result["completed_work"] == []
    assert result["unresolved_issues"] == [
        "The feature is not finished"
    ]
