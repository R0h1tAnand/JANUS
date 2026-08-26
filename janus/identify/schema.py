"""Machine-readable schema for the Janus attack atlas.

Every attack card in ``janus/identify/atlas/`` validates against :class:`AttackCard`.
Cards are not documentation: the generator compiles simulated cards into injectors and
the defender maps ``observables`` onto concrete features, so the vocabulary below is a
shared contract across all three pillars.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from janus.identify.signals import OBSERVABLE_SIGNALS, SignalFamily


class Rail(StrEnum):
    """Payment rail the attack lands on.

    India rails are modelled first-class because the fraud economics differ sharply from
    card rails: UPI is real-time, irrevocable and push-based, which removes the
    chargeback safety net that card fraud detection implicitly leans on.
    """

    UPI_P2P = "upi_p2p"
    UPI_P2M = "upi_p2m"
    UPI_COLLECT = "upi_collect"
    UPI_MANDATE = "upi_mandate"
    UPI_LITE = "upi_lite"
    IMPS = "imps"
    NEFT = "neft"
    RTGS = "rtgs"
    AEPS = "aeps"
    CARD_CNP = "card_cnp"
    CARD_CP = "card_cp"
    CARD_TOKEN = "card_token"
    CARD_ATM = "card_atm"
    WALLET_PPI = "wallet_ppi"
    CROSS_BORDER = "cross_border"


class GenAIEnabler(StrEnum):
    """The specific generative capability that makes the attack cheaper/faster/scalable.

    This axis is what separates a Janus card from a generic fraud typology: each card must
    name what GenAI actually changed, not merely assert that AI was involved.
    """

    VOICE_CLONE = "voice_clone"
    DEEPFAKE_VIDEO = "deepfake_video"
    LLM_DIALOGUE = "llm_dialogue"
    LLM_CONTENT = "llm_content"
    DOC_FORGERY = "doc_forgery"
    PERSONA_SYNTHESIS = "persona_synthesis"
    AGENTIC_AUTOMATION = "agentic_automation"
    ADVERSARIAL_ML = "adversarial_ml"
    CODE_GEN = "code_gen"
    LOCALISATION = "localisation"


class Surface(StrEnum):
    """Where the human or system is engaged."""

    VOICE_CALL = "voice_call"
    IVR = "ivr"
    SMS = "sms"
    CHAT_APP = "chat_app"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    WEB = "web"
    MOBILE_APP = "mobile_app"
    QR_PHYSICAL = "qr_physical"
    MERCHANT_CHATBOT = "merchant_chatbot"
    VIDEO_KYC = "video_kyc"
    MARKETPLACE = "marketplace"
    SCREEN_SHARE = "screen_share"
    API_DIRECT = "api_direct"


class Phase(StrEnum):
    """Kill-chain phase, ordered. Used to render the ATT&CK-style matrix columns."""

    RECON = "recon"
    RESOURCE_DEV = "resource_dev"
    PRETEXT = "pretext"
    TRUST_BUILD = "trust_build"
    CREDENTIAL_ACCESS = "credential_access"
    INSTRUMENT = "instrument"
    EVASION = "evasion"
    PERSIST = "persist"
    MONETIZE = "monetize"
    LAUNDER = "launder"


PHASE_ORDER: list[Phase] = list(Phase)


class Monetization(StrEnum):
    """How value actually leaves the victim."""

    PUSH_PAYMENT = "push_payment"
    CNP_PURCHASE = "cnp_purchase"
    CASH_OUT = "cash_out"
    MULE_TRANSFER = "mule_transfer"
    CRYPTO_OFFRAMP = "crypto_offramp"
    GOODS_RESALE = "goods_resale"
    REFUND_ABUSE = "refund_abuse"
    CREDIT_BUSTOUT = "credit_bustout"
    GIFT_CARD = "gift_card"
    SUBSCRIPTION_SIPHON = "subscription_siphon"


class Status(StrEnum):
    """Maturity of the vector in the wild, as of the atlas revision date."""

    THEORETICAL = "theoretical"
    EMERGING = "emerging"
    ESTABLISHED = "established"


class KillChainStep(BaseModel):
    """One ordered step in how the attack unfolds."""

    phase: Phase
    description: str = Field(min_length=10)
    genai_used: bool = False

    model_config = {"extra": "forbid"}


class VictimProfile(BaseModel):
    """Who the attack selects for. Drives victim sampling in the simulator."""

    age_bands: list[str] = Field(default_factory=list)
    digital_tenure: str | None = None
    segment: str | None = None
    notes: str | None = None

    model_config = {"extra": "forbid"}


class AttackCard(BaseModel):
    """A single, simulatable payment-fraud technique.

    ``simulated=True`` cards MUST name an ``injector``; the atlas loader enforces that the
    named injector actually resolves, so the atlas can never overstate coverage.
    """

    id: str = Field(pattern=r"^VY-[A-Z]{3,6}-\d{3}$")
    name: str = Field(min_length=5, max_length=120)
    summary: str = Field(min_length=40)

    rails: list[Rail] = Field(min_length=1)
    genai_enablers: list[GenAIEnabler] = Field(min_length=1)
    surfaces: list[Surface] = Field(default_factory=list)
    monetization: list[Monetization] = Field(min_length=1)

    kill_chain: list[KillChainStep] = Field(min_length=2)
    observables: list[str] = Field(min_length=1)
    mitigations: list[str] = Field(default_factory=list)

    victim_profile: VictimProfile = Field(default_factory=VictimProfile)

    status: Status = Status.EMERGING
    severity: int = Field(ge=1, le=5, description="Loss magnitude per successful attack")
    scalability: int = Field(ge=1, le=5, description="How well GenAI multiplies attempt volume")
    detectability: int = Field(
        ge=1, le=5, description="1 = trivially caught by rules today, 5 = near-invisible"
    )

    simulated: bool = False
    injector: str | None = Field(
        default=None, description="Dotted path under janus.generate.injectors"
    )
    references: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("observables")
    @classmethod
    def _observables_are_registered(cls, v: list[str]) -> list[str]:
        """Every observable must be a registered signal.

        This is the contract that keeps Pillar 1 and Pillar 3 honest: a card cannot claim a
        detectable signal unless ``janus.identify.signals`` says which family of feature is
        responsible for computing it.
        """
        bad = [o for o in v if o != o.lower().replace(" ", "_") or not o.isascii()]
        if bad:
            raise ValueError(f"observables must be lowercase snake_case ascii: {bad}")
        unregistered = [o for o in v if o not in OBSERVABLE_SIGNALS]
        if unregistered:
            raise ValueError(
                f"unregistered observables {unregistered}; add them to "
                "janus/identify/signals.py so a feature family owns them"
            )
        return v

    @field_validator("kill_chain")
    @classmethod
    def _distinct_phases(cls, v: list[KillChainStep]) -> list[KillChainStep]:
        """Kill chains need real progression, but not chronological authoring order.

        Attackers routinely build tooling before reconnaissance, and evasion is interleaved
        rather than sequential, so ordering is not enforced - only that the card describes
        more than one distinct phase.
        """
        if len({s.phase for s in v}) < 2:
            raise ValueError("kill_chain must span at least two distinct phases")
        return v

    @model_validator(mode="after")
    def _simulated_needs_injector(self) -> AttackCard:
        if self.simulated and not self.injector:
            raise ValueError(f"{self.id}: simulated=true requires an injector")
        if self.injector and not self.simulated:
            raise ValueError(f"{self.id}: injector set but simulated=false")
        return self

    @property
    def risk_score(self) -> float:
        """Prioritisation heuristic: impact x reach x stealth, normalised to 0-1.

        Deliberately simple and transparent - this drives which cards we simulate first,
        and a judge should be able to recompute it by hand from the card.
        """
        return round((self.severity * self.scalability * self.detectability) / 125, 3)

    @property
    def signal_families(self) -> set[SignalFamily]:
        """The feature families this card's observables require the defence to implement."""
        return {OBSERVABLE_SIGNALS[o] for o in self.observables}

    @property
    def family(self) -> str:
        """Short family key used to group cards and to label LOAO evaluation folds."""
        return self.id.split("-")[1].lower()
