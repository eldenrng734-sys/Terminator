from app.recommend.models import Candidate, ScoredCandidate

MIN_VOTE_COUNT = 200
MIN_AUDIENCE_SCORE = 7.0
MIN_RT_SCORE = 70


def passes_quality_bar(candidate: Candidate) -> bool:
    if candidate.vote_count < MIN_VOTE_COUNT:
        return False
    if candidate.vote_average < MIN_AUDIENCE_SCORE:
        return False
    return candidate.rt_score is None or candidate.rt_score >= MIN_RT_SCORE


def overlap_score(
    candidate: Candidate, source: Candidate
) -> tuple[float, list[str]]:
    score = candidate.vote_average - MIN_AUDIENCE_SCORE
    reasons: list[str] = []

    shared_genres = set(candidate.genre_ids) & set(source.genre_ids)
    if shared_genres:
        score += len(shared_genres) * 2.0
        reasons.append("shares genre")

    shared_keywords = set(candidate.keyword_ids) & set(source.keyword_ids)
    if shared_keywords:
        score += len(shared_keywords) * 1.5
        reasons.append("similar themes")

    shared_cast = set(candidate.cast_ids) & set(source.cast_ids)
    if shared_cast:
        score += min(len(shared_cast), 3) * 1.0
        reasons.append("shares cast")

    if candidate.year is not None and source.year is not None:
        if abs(candidate.year - source.year) <= 5:
            score += 0.5
            reasons.append("same era")

    return score, reasons


def filter_and_rank(
    pool: list[Candidate],
    source: Candidate | None = None,
    limit: int = 5,
) -> list[ScoredCandidate]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    unique: dict[tuple[str, int], Candidate] = {}
    for candidate in pool:
        if source is not None and (
            candidate.media_type == source.media_type
            and candidate.tmdb_id == source.tmdb_id
        ):
            continue
        unique.setdefault((candidate.media_type, candidate.tmdb_id), candidate)

    scored: list[ScoredCandidate] = []
    for candidate in unique.values():
        if not passes_quality_bar(candidate):
            continue
        if source is not None:
            score, reasons = overlap_score(candidate, source)
        else:
            score = candidate.vote_average - MIN_AUDIENCE_SCORE
            reasons = []
        if candidate.imdb_rating is not None:
            score += max(0.0, candidate.imdb_rating - 7.0) * 0.25
        scored.append(
            ScoredCandidate(
                **candidate.model_dump(), score=score, reasons=reasons
            )
        )

    scored.sort(key=lambda candidate: (-candidate.score, -candidate.vote_count))
    return scored[:limit]