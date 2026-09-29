export interface UserCreate {
  email: string;
  username: string;
  password: string;
}

export interface UserRead {
  id: number;
  email: string;
  username: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserProfile {
  id: number;
  email: string;
  username: string;
}
