function assertValueData_(x, name)
% ASSERTVALUEDATA_  Artifacts follow the same "value data only" rule as the Core records: one shared definition,
% OceanAgents.Core.assertValueData (a handle inside an artifact would be shared by every policy built from it).
OceanAgents.Core.assertValueData(x, name);
end
