export interface VehicleProfile {
  color: string | null;
  body_type: string | null;
  features: string[];
  brand: string | null;
  model: string | null;
}

export interface VehicleMatch {
  camera_id: string;
  timestamp: string;
  image_path: string;
  score: number;
}

export interface IdentifyResponse {
  profile: VehicleProfile;
  matches: VehicleMatch[];
}

export interface HealthResponse {
  status: string;
  index_size: number;
}
