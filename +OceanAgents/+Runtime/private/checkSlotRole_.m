function checkSlotRole_(host, slotRole, slotName)
% CHECKSLOTROLE_  A host that declares a role must play a slot of exactly that role (else Runtime:RosterMismatch).
% A host left at the default role "default" is role-agnostic and takes the slot's role; either way the runner then
% asks the mind (and so any imported policy artifact) to checkCompatible against the SLOT's role and spaces.
if host.Role ~= "default" && host.Role ~= slotRole
    fail_('RosterMismatch', 'Slot %s has role "%s"; host %s declares role "%s".', slotName, slotRole, ...
        host.AgentID, host.Role);
end
end
