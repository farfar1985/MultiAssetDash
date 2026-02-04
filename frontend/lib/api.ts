import type {
  ApiResponse,
  PaginatedResponse,
  Signal,
  DashboardMetrics,
  PersonaId,
  AssetId,
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;

    const defaultHeaders: HeadersInit = {
      "Content-Type": "application/json",
    };

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return {
        data,
        success: true,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        data: null as T,
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Dashboard endpoints
  async getDashboardMetrics(persona: PersonaId): Promise<ApiResponse<DashboardMetrics>> {
    return this.request<DashboardMetrics>(`/api/dashboard/${persona}/metrics`);
  }

  // Signal endpoints
  async getSignals(
    persona: PersonaId,
    params?: {
      asset?: AssetId;
      page?: number;
      pageSize?: number;
    }
  ): Promise<PaginatedResponse<Signal>> {
    const searchParams = new URLSearchParams();
    if (params?.asset) searchParams.set("asset", params.asset);
    if (params?.page) searchParams.set("page", params.page.toString());
    if (params?.pageSize) searchParams.set("pageSize", params.pageSize.toString());

    const query = searchParams.toString();
    const endpoint = `/api/signals/${persona}${query ? `?${query}` : ""}`;

    const response = await this.request<Signal[]>(endpoint);
    return {
      ...response,
      data: response.data || [],
      pagination: {
        page: params?.page || 1,
        pageSize: params?.pageSize || 20,
        total: 0,
        totalPages: 0,
      },
    };
  }

  async getSignalById(id: string): Promise<ApiResponse<Signal>> {
    return this.request<Signal>(`/api/signals/${id}`);
  }

  // Asset endpoints
  async getAssetDetails(
    assetId: AssetId,
    persona: PersonaId
  ): Promise<ApiResponse<{
    asset: AssetId;
    currentPrice: number;
    signals: Signal[];
    metrics: Record<string, number>;
  }>> {
    return this.request(`/api/assets/${assetId}?persona=${persona}`);
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; version: string }>> {
    return this.request("/health");
  }
}

export const apiClient = new ApiClient();
export default apiClient;
