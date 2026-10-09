const OpenAI = require('openai');
const createOpenAIImageTools = require('~/app/clients/tools/structured/OpenAIImageTools');

jest.mock('openai');
jest.mock('@librechat/data-schemas', () => ({
  logger: {
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

jest.mock('@librechat/api', () => ({
  logAxiosError: jest.fn(),
  oaiToolkit: {
    image_gen_oai: {
      name: 'image_gen_oai',
      description: 'Generate an image',
      schema: {},
    },
    image_edit_oai: {
      name: 'image_edit_oai',
      description: 'Edit an image',
      schema: {},
    },
  },
  extractBaseURL: jest.fn((url) => url),
  getProxyDispatcher: jest.fn(() => undefined),
  applyAxiosProxyConfig: jest.fn(),
}));

jest.mock('~/server/services/Files/strategies', () => ({
  getStrategyFunctions: jest.fn(),
}));

jest.mock('~/models', () => ({
  getFiles: jest.fn().mockResolvedValue([]),
}));

/** Mirrors the SDK's `APIPromise`: the tool reads the body and the headers with `withResponse()`. */
const apiResponse = (data, headers = {}) => ({
  withResponse: () => Promise.resolve({ data, response: { headers: new Headers(headers) } }),
});

const imageResponse = () => apiResponse({ data: [{ b64_json: 'base64-encoded-image-data' }] });

describe('OpenAIImageTools - IMAGE_GEN_OAI_MODEL environment variable', () => {
  let originalEnv;

  beforeEach(() => {
    jest.clearAllMocks();
    originalEnv = { ...process.env };

    process.env.IMAGE_GEN_OAI_API_KEY = 'test-api-key';

    OpenAI.mockImplementation(() => ({
      images: {
        generate: jest.fn().mockImplementation(imageResponse),
      },
    }));
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('should use default model "gpt-image-1" when IMAGE_GEN_OAI_MODEL is not set', async () => {
    delete process.env.IMAGE_GEN_OAI_MODEL;

    const [imageGenTool] = createOpenAIImageTools({
      isAgent: true,
      override: false,
      req: { user: { id: 'test-user' } },
    });

    const mockGenerate = jest.fn().mockImplementation(imageResponse);

    OpenAI.mockImplementation(() => ({
      images: {
        generate: mockGenerate,
      },
    }));

    await imageGenTool.func({ prompt: 'test prompt' });

    expect(mockGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'gpt-image-1',
      }),
      expect.any(Object),
    );
  });

  it('should use "gpt-image-1.5" when IMAGE_GEN_OAI_MODEL is set to "gpt-image-1.5"', async () => {
    process.env.IMAGE_GEN_OAI_MODEL = 'gpt-image-1.5';

    const mockGenerate = jest.fn().mockImplementation(imageResponse);

    OpenAI.mockImplementation(() => ({
      images: {
        generate: mockGenerate,
      },
    }));

    const [imageGenTool] = createOpenAIImageTools({
      isAgent: true,
      override: false,
      req: { user: { id: 'test-user' } },
    });

    await imageGenTool.func({ prompt: 'test prompt' });

    expect(mockGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'gpt-image-1.5',
      }),
      expect.any(Object),
    );
  });

  it('should use custom model name from IMAGE_GEN_OAI_MODEL environment variable', async () => {
    process.env.IMAGE_GEN_OAI_MODEL = 'custom-image-model';

    const mockGenerate = jest.fn().mockImplementation(imageResponse);

    OpenAI.mockImplementation(() => ({
      images: {
        generate: mockGenerate,
      },
    }));

    const [imageGenTool] = createOpenAIImageTools({
      isAgent: true,
      override: false,
      req: { user: { id: 'test-user' } },
    });

    await imageGenTool.func({ prompt: 'test prompt' });

    expect(mockGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'custom-image-model',
      }),
      expect.any(Object),
    );
  });
});

describe('OpenAIImageTools - successful response with no image data', () => {
  let originalEnv;

  const empty = () => apiResponse({ data: [] }, { 'x-litellm-call-id': 'call-123' });

  const createGenTool = (mockGenerate) => {
    OpenAI.mockImplementation(() => ({
      images: {
        generate: mockGenerate,
      },
    }));
    const [imageGenTool] = createOpenAIImageTools({
      isAgent: true,
      override: false,
      req: { user: { id: 'test-user' } },
    });
    return imageGenTool;
  };

  beforeEach(() => {
    jest.clearAllMocks();
    originalEnv = { ...process.env };
    process.env.IMAGE_GEN_OAI_API_KEY = 'test-api-key';
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('should retry once and return the image when the first response is empty', async () => {
    const mockGenerate = jest
      .fn()
      .mockImplementationOnce(empty)
      .mockImplementationOnce(imageResponse);

    const [content, artifact] = await createGenTool(mockGenerate).func({ prompt: 'test prompt' });

    expect(mockGenerate).toHaveBeenCalledTimes(2);
    expect(mockGenerate.mock.calls[1][0]).toEqual(mockGenerate.mock.calls[0][0]);
    expect(artifact.content[0].image_url.url).toContain('base64-encoded-image-data');
    expect(content[0].text).toContain('generated_image_id');
  });

  it('should return a clear message instead of throwing when both responses are empty', async () => {
    const { logger } = require('@librechat/data-schemas');
    const mockGenerate = jest.fn().mockImplementation(empty);

    const [content, artifact] = await createGenTool(mockGenerate).func({ prompt: 'test prompt' });

    expect(mockGenerate).toHaveBeenCalledTimes(2);
    expect(content).toContain('returned no image');
    expect(artifact).toEqual({});
    expect(logger.warn).toHaveBeenLastCalledWith(
      expect.stringContaining('no image data'),
      expect.objectContaining({ request_id: 'call-123', prompt_length: 11 }),
    );
  });

  it('should treat a response without a data array as empty', async () => {
    const mockGenerate = jest.fn().mockImplementation(() => apiResponse({}));

    const [content] = await createGenTool(mockGenerate).func({ prompt: 'test prompt' });

    expect(mockGenerate).toHaveBeenCalledTimes(2);
    expect(content).toContain('returned no image');
  });

  it('should not retry when the first response has an image', async () => {
    const mockGenerate = jest.fn().mockImplementation(imageResponse);

    await createGenTool(mockGenerate).func({ prompt: 'test prompt' });

    expect(mockGenerate).toHaveBeenCalledTimes(1);
  });
});
