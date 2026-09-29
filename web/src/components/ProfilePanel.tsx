import type { VehicleProfile } from "../types";

const FIELDS: Array<[label: string, key: keyof VehicleProfile]> = [
  ["Марка", "brand"],
  ["Модель", "model"],
  ["Цвет", "color"],
  ["Кузов", "body_type"],
];

export function ProfilePanel({ profile }: { profile: VehicleProfile }) {
  return (
    <div className="profile">
      {FIELDS.map(([label, key]) => (
        <div className="profile-row" key={key}>
          <span className="k">{label}</span>
          <span className="v">{(profile[key] as string | null) ?? "—"}</span>
        </div>
      ))}
      {profile.features.length > 0 && (
        <div className="profile-row">
          <span className="k">Особенности</span>
          <div className="chips">
            {profile.features.map((f) => (
              <span className="chip" key={f}>
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
