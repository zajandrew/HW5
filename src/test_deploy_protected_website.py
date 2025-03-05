from deploy_protected_website import (
    DEPLOYED_PROTECTED_WEBSITE,
    URL_TO_PROTECTED_WEBSITE,
    CLASSMATE_IS_AUTHORIZED,
)


def test_deploy_protected_website():
    assert DEPLOYED_PROTECTED_WEBSITE
    assert URL_TO_PROTECTED_WEBSITE != ""


def test_authorized_users():
    assert CLASSMATE_IS_AUTHORIZED
