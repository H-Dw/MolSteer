from pydantic import ValidationError
from molsteer.agents.config import StrictModel
from molsteer.agents.loop import _validation_feedback


def test_schema_feedback_identifies_missing_fields_without_echoing_values():
    class Contract(StrictModel):
        direction_id: str
        observable_id: str
    try:
        Contract.model_validate({'direction_id':'private-input-must-not-echo'})
    except ValidationError as exc:
        feedback = _validation_feedback(exc)
    assert feedback == {'validation_errors':[{'path':['observable_id'],'rule':'missing'}]}
    assert 'private-input-must-not-echo' not in str(feedback)


def test_unknown_exception_messages_do_not_enter_model_feedback():
    assert _validation_feedback(ValueError('credential=must-not-leak')) == {}
    assert _validation_feedback(RuntimeError('provider-response-must-not-leak')) == {}


def test_fixed_public_contract_rules_can_be_repaired():
    message='Each direction requires its own actual local retrieval'
    assert _validation_feedback(ValueError(message)) == {'validation_rule':message}
