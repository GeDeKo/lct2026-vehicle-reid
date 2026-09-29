import { assetUrl } from "../api";
import { formatTimestamp } from "../format";
import type { VehicleMatch } from "../types";

export function MatchTile({ match, index }: { match: VehicleMatch; index: number }) {
  const pct = Math.round(match.score * 100);
  return (
    <div className="match-tile" style={{ animationDelay: `${Math.min(index * 35, 350)}ms` }}>
      <div className="match-frame">
        {match.image_path && <img src={assetUrl(match.image_path)} alt="" loading="lazy" />}
        <span className="match-tag">
          {match.camera_id} · {formatTimestamp(match.timestamp)}
        </span>
      </div>
      <div className="match-meta">
        <div className="meter">
          <div className="meter-track">
            <div className="meter-fill" style={{ width: `${pct}%` }} />
          </div>
          <span className="meter-value">{pct}%</span>
        </div>
      </div>
    </div>
  );
}
